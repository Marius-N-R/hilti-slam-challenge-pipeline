import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. Setup Model
model = build_sam3_image_model()
processor = Sam3Processor(model)



# 2. Load Image
image_path = "./pictures/input/input.png"
image = Image.open(image_path).convert("RGB")
W, H = image.size

# crop out center portion

# Cast to int immediately
left = int(W * 0.25)
right = int(W * 0.75)
top = int(H * 0)
bottom = int(H * 1)

# Now these variables are integers, safe for crop
center_img = image.crop((left, top, right, bottom))

result_img = Image.new(image.mode, (W, H), (0, 0, 0))

# Safe for paste as well
result_img.paste(center_img, (left, top))
image = result_img



# Load Bounding boxes (already in cx, cy, w, h normalized 0-1)
bbxes = np.load("./pictures/GroundingDINO_out/bb.npy").astype(np.float32) 

filtered_bbxes = []
for bb_idx, bb in enumerate(bbxes):
    # Filter by size (normalized)
    if bb[2] > 0.6 or bb[3] > 0.6: continue
    if bb[2] < 0.025 or bb[3] < 0.025: continue
    
    # Simple similarity filter
    is_duplicate = False
    for accepted_bb in filtered_bbxes:
        if np.linalg.norm(bb[:2] - accepted_bb[:2]) < 0.05: # Too close to another box
            is_duplicate = True
            break
    if not is_duplicate:
        filtered_bbxes.append(bb)

print(f"[INFO] Processing {len(filtered_bbxes)} filtered boxes...")

# 3. Run Inference
with torch.autocast("cuda", dtype=torch.bfloat16):
    # This one DOES return the state, so we assign it
    inference_state = processor.set_image(image)
    
    # These usually modify the state IN-PLACE, so do NOT re-assign
    processor.reset_all_prompts(inference_state)

    for box in filtered_bbxes:
        # Pass the state, but don't overwrite it with the function's return value
        processor.add_geometric_prompt(
            box=box.tolist(), 
            label=True, # Must be Boolean True, not "positive"
            state=inference_state
        )
    
    # Finally, trigger the model using the updated state
    output = processor._forward_grounding(state=inference_state)

# 4. Extract results
masks = output["masks"]
print(f"Generated {len(masks)} masks.")

#################################### Visualization ####################################

def show_masks_on_image(raw_image, masks):
    plt.figure(figsize=(12, 8))
    plt.imshow(raw_image)
    mask_img = np.zeros((raw_image.size[1], raw_image.size[0],3), dtype=np.uint8)
    if len(masks) > 0:
        if torch.is_tensor(masks):
            masks = masks.cpu().float().numpy()
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Convert all to binary
        binary_masks = [(mask > 0.5).astype(np.uint8) for mask in masks]

        # Pick the largest mask
        largest_mask = max(binary_masks, key=lambda m: m.sum())

        # Save only that one
        mask_img = largest_mask.reshape(H, W, 1) * 255
        np.save("./pictures/output/windows_masks.npy", mask_img)
        
    #     if torch.is_tensor(masks):
    #         masks = masks.cpu().float().numpy()
            
    #     # Handle shape [N, 1, H, W]
    #     if masks.ndim == 4:
    #         masks = masks.squeeze(1)

    #     for i, mask in enumerate(masks):
    #         # Convert probability map to binary mask
    #         mask_binary = (mask > 0.5).astype(np.uint8)
            
    #         # Create color overlay
    #         color = np.concatenate([plt.cm.tab10(i % 10)[:3], [0.5]]) 
    #         h, w = mask_binary.shape
    #         mask_image = mask_binary.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #         plt.gca().imshow(mask_image)
    #         mask_img += mask_binary.reshape(h,w,1) * 255

            
    # plt.axis('off')
    # plt.savefig("./pictures/output/windows_segmented.png", bbox_inches='tight', pad_inches=0)
    # print("Saved windows_segmented.png")
    # np.save("./pictures/output/windows_masks.npy", mask_img)
    # print(mask_img)

show_masks_on_image(image, masks)