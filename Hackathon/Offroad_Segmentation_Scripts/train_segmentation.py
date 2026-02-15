
import os
# Set MPS env vars BEFORE importing torch
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import evaluate
from PIL import Image

# =================================================================================================
# 1. CONFIGURATION
# =================================================================================================

# Dataset Paths
DATASET_ROOT = "../Offroad_Segmentation_Training_Dataset"
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "train", "Color_Images")
TRAIN_MASK_DIR = os.path.join(DATASET_ROOT, "train", "Segmentation")
VAL_IMG_DIR = os.path.join(DATASET_ROOT, "val", "Color_Images")
VAL_MASK_DIR = os.path.join(DATASET_ROOT, "val", "Segmentation")

# Model Settings
PRETRAINED_MODEL_NAME = "nvidia/mit-b0" 
OUTPUT_DIR = "./segformer_offroad_output"

# Hyperparameters
BATCH_SIZE = 1 # Reduced to 1 to fix MPS "stride" bug
EPOCHS = 15
LEARNING_RATE = 0.00006
# IMAGE_SIZE = (512, 512) 

# Label Mapping (Raw ID -> Train ID)
ID2LABEL = {0: "Obstacle", 1: "Landscape", 2: "Grass", 3: "Rough"}
LABEL2ID = {"Obstacle": 0, "Landscape": 1, "Grass": 2, "Rough": 3}
NUM_LABELS = len(ID2LABEL)

# =================================================================================================
# 2. DATASET CLASS
# =================================================================================================

class OffroadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.images = [i for i in self.images if i in self.masks]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) 
        
        new_mask = np.zeros_like(mask, dtype=np.uint8) 
        new_mask[mask == 7100] = 1
        new_mask[mask == 200] = 2
        new_mask[(mask == 300) | (mask == 550)] = 3
        
        mask_pil = Image.fromarray(new_mask)
        # Processor handles resizing and normalization
        encoded_inputs = self.processor(image, mask_pil, return_tensors="pt")
        
        # Squeeze batch dim from processor output
        flat_inputs = {k: v.squeeze() for k, v in encoded_inputs.items()}
        return flat_inputs

# Custom Collator to enforce contiguous memory on the batch
def contiguous_collate_fn(features):
    batch = {}
    first = features[0]
    for key in first.keys():
        # Stack
        tensor_list = [f[key] for f in features]
        stacked = torch.stack(tensor_list)
        # Force contiguous
        batch[key] = stacked.contiguous()
    return batch

# =================================================================================================
# 3. METRICS
# =================================================================================================

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits).contiguous()
        
        logits_tensor = torch.nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        
        pred_labels = logits_tensor.detach().cpu().numpy()
        
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=NUM_LABELS,
            ignore_index=255, 
            reduce_labels=False,
        )
        
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        result = {
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"],
            "accuracy_landscape": per_category_accuracy[1] if len(per_category_accuracy) > 1 else 0
        }
        return result

# =================================================================================================
# 4. MAIN TRAINING
# =================================================================================================

def main():
    use_mps = False
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        use_mps = True
        print("Using MPS (Apple Silicon Acceleration)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 1. Setup Image Processor
    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
    processor.do_reduce_labels = False 
    processor.size = {"height": 512, "width": 512}

    # 2. Datasets
    train_ds = OffroadDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, processor)
    val_ds = OffroadDataset(VAL_IMG_DIR, VAL_MASK_DIR, processor)
    
    print(f"Training Samples: {len(train_ds)}")
    print(f"Validation Samples: {len(val_ds)}")

    # 3. Model
    model = AutoModelForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True 
    )
    model.to(device)

    # 4. Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4, # Simulate larger batch
        save_total_limit=2,
        eval_strategy="epoch", 
        save_strategy="epoch",
        logging_steps=50,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",
        dataloader_pin_memory=False, 
        dataloader_num_workers=0, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=contiguous_collate_fn, 
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving best model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "best_model"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
    print("Done!")

if __name__ == "__main__":
    main()
