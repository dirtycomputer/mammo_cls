import json
import os
from PIL import Image
import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torchvision import transforms
from accelerate import Accelerator

import numpy as np

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from timm.models.layers import to_2tuple

def compute_metrics(labels, preds):
    # 将概率转换为二进制预测
    preds_binary = (preds > 0.5).astype(np.int16)

    # 计算 TP, TN, FP, FN
    TP = ((preds_binary == 1) & (labels == 1)).sum().item()
    TN = ((preds_binary == 0) & (labels == 0)).sum().item()
    FP = ((preds_binary == 1) & (labels == 0)).sum().item()
    FN = ((preds_binary == 0) & (labels == 1)).sum().item()

    # 计算指标
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    # 计算 AUROC
    auroc = roc_auc_score(labels, preds)
    wandb.log({
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1 Score": f1,
        "AUROC": auroc
    })
    preds_2cls = [[1 - prob, prob] for prob in preds]
    preds_int = np.where(preds >= 0.5, 1, 0).astype(int)
    class_names = ["positive","negative"]
    wandb.log({
        "roc" : wandb.plot.roc_curve(labels, preds_2cls, labels=class_names),
        "pr" : wandb.plot.pr_curve(labels, preds_2cls, labels=class_names),
        "confusion_matrix" : wandb.plot.confusion_matrix(
            probs=None,y_true=labels,
            preds=preds_int,class_names=class_names)
    })


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x
    
class FixedFFT(nn.Module):
    def __init__(self, in_chans=3, embed_dim=2048, img_size=224, patch_size=14):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        assert self.img_size % self.patch_size == 0, "img_size is not divisible by patch_size"
        self.num_tokens = (self.img_size//self.patch_size) * (self.img_size//self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        # B, H/p*W/p+1, dim
        batch_size, seq_len, _ = x.size()
        freq = torch.fft.fft(x, dim=(-2), n=self.num_tokens, norm="ortho")
        x = torch.abs(freq)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

# PyTorch 数据集类
class MedicalImageDataset(Dataset):
    def __init__(self, image_folder_path, model_name="1B", max_length=500, transform=None):
        json_path = image_folder_path.replace("image","filter_cap.json")
        with open(json_path) as f:
            self.json_data = json.load(f)
        self.image_folder_path = image_folder_path
        self.transform = transform
        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.json_data["annotations"])

    def __getitem__(self, idx):
        item = self.json_data["annotations"][idx]
        image_id = item["image_id"]
        caption = item["caption"]
        label = 1 if "恶性" in caption else 0
        caption = caption.split("结节为")[0]
        image_path = os.path.join(self.image_folder_path, image_id + ".jpg")
        image = Image.open(image_path)
        tokenized_text = self.llama_tokenizer(caption, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.max_length)
        input_ids = tokenized_text['input_ids'].squeeze()

        if self.transform:
            image = self.transform(image)

        return image, input_ids, label

# 定义图像转换
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    ])

class fftVLM(nn.Module):
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=2048, model_name="1B", mode="mixed"):
        super().__init__()
        self.mode = mode
        self.llama_config = AutoConfig.from_pretrained(model_name)
        self.llama_model_classifier = AutoModelForSequenceClassification.from_config(self.llama_config)
        self.llama_model_classifier.config.pad_token_id = self.llama_model_classifier.config.eos_token_id
        if self.mode in ["image", "mixed"]:
            self.vision_encoder = FixedFFT(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            self.ln_vision = nn.LayerNorm(self.vision_encoder.num_features)
            self.llama_proj = nn.Linear(self.vision_encoder.num_features, self.llama_model_classifier.config.hidden_size)
        print(self.llama_config)
        print(self.llama_model_classifier)
        for name, param in self.llama_model_classifier.named_parameters():
            if "score" not in name:
                param.requires_grad = False
        
    
    def forward(self, images, input_ids, labels):
        if self.mode in ["image", "mixed"]:
            image_embeds = self.ln_vision(self.vision_encoder(images))
            bs, pn, hs = image_embeds.shape
            aligned_image_embeds = self.llama_proj(image_embeds)
        text_embeds = self.llama_model_classifier.base_model.embed_tokens(input_ids)
        if self.mode in ["image"]:
            mixed_embeds = aligned_image_embeds
        elif self.mode in ["text"]:
            mixed_embeds = text_embeds
        elif self.mode in ["mixed"]:
            mixed_embeds = torch.concat((aligned_image_embeds, text_embeds), dim=1)
        attention_masks = torch.ones_like(mixed_embeds)
        outputs =self.llama_model_classifier(
                inputs_embeds=mixed_embeds,
                attention_mask=attention_masks,
                return_dict=True,
                labels=labels,
            )
        return outputs
        
        
        
def main():
    wandb.init(project="minigpt")
    wandb.save("classifier.py")
    epochs = 50
    gradient_accumulation_steps = 8
    accelerator = Accelerator()
    train_image_path = "/data/datasets/HMBM/samples_train/image"
    test_image_path = "/data/datasets/HMBM/samples_test/image"
    model_name = "/data/llm_weights/1B"
    train_dataset = MedicalImageDataset(train_image_path, transform=transform, model_name=model_name)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataset = MedicalImageDataset(test_image_path, transform=transform, model_name=model_name)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    if accelerator.is_main_process:
        print("train dataset length:",len(train_dataset))
        print("test dataset length:",len(test_dataset))
    
    model = accelerator.prepare(fftVLM(model_name=model_name, mode="mixed"))
    optimizer = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.95), weight_decay=1e-5)
    optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        optimizer, train_dataloader, test_dataloader
        )
    
    for epoch in range(epochs):
        model.train()
        if accelerator.is_main_process:
            print("*"*10)
            print("epoch:",epoch)
        for index,(images, captions, labels) in enumerate(train_dataloader):
            loss = model(images, captions, labels)["loss"]
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            if (index+1) % gradient_accumulation_steps == 0:
                wandb.log({"loss":loss})
                optimizer.step()
                optimizer.zero_grad()

        model.eval()        
        with torch.no_grad():
            all_logits = []
            all_labels = []
            for images, captions, labels in test_dataloader:
                logits = model(images, captions, labels)["logits"]
                logits = accelerator.gather_for_metrics(logits)
                labels = accelerator.gather_for_metrics(labels)

                all_logits.append(logits)
                all_labels.append(labels)
            
            if accelerator.is_main_process:
                all_logits = F.softmax(torch.concat(all_logits),dim=1)[:,1].cpu().numpy()
                all_labels = torch.concat(all_labels, dim=0).squeeze().cpu().numpy()
                print(all_logits.shape, all_labels.shape)
                compute_metrics(all_labels, all_logits)
                
if __name__ == "__main__":
    main()      
