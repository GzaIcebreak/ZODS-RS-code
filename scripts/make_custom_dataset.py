import os
import json
import requests
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- CONFIGURATION ---
annotation_file = 'data/coco/annotations/instances_train2017.json'
output_folder = 'data/my_custom_dataset'

selection = {
    "reference": {
        "bird": [429819],
        "boat": [101435]
    },
    "targets": [98636, 361948, 456065, 42279, 517410, 439274, 407180, 459673, 481301, 460598]
}

coco_images_url = "http://images.cocodataset.org/train2017/{}.jpg"

# --- CREATE OUTPUT FOLDERS ---
images_dir = os.path.join(output_folder, "images")
annotations_dir = os.path.join(output_folder, "annotations")
vis_dir = os.path.join(annotations_dir, "references_visualisations")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# --- LOAD COCO DATA ---
coco = COCO(annotation_file)
category_name_to_id = {cat['name']: cat['id'] for cat in coco.loadCats(coco.getCatIds())}

# --- PROCESS REFERENCES ---
new_images = []
new_annotations = []
new_categories = []
used_category_ids = set()
used_image_ids = set()
new_ann_id = 1

for class_name, image_ids in selection["reference"].items():
    cat_id = category_name_to_id[class_name]
    for image_id in image_ids:
        ann_ids = coco.getAnnIds(imgIds=image_id, catIds=cat_id)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            continue

        img_info = coco.loadImgs(image_id)[0]
        new_images.append(img_info)

        for ann in anns:
            ann = ann.copy()  # Make sure we don't modify the original COCO dict
            ann.pop("segmentation", None)  # Remove segmentation if present
            ann['id'] = new_ann_id
            new_ann_id += 1
            new_annotations.append(ann)

        used_category_ids.add(cat_id)
        used_image_ids.add(image_id)

        # Download and visualize
        img_filename = coco.loadImgs(image_id)[0]['file_name']
        img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_path):
            url = coco_images_url.format(str(image_id).zfill(12))
            img_data = requests.get(url).content
            with open(img_path, 'wb') as f:
                f.write(img_data)

        # Visualisation
        image = Image.open(img_path)
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for ann in anns:
            bbox = ann['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        vis_path = os.path.join(vis_dir, f"{class_name}_{image_ids.index(image_id)+1}.jpg")
        plt.axis('off')
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0)
        plt.close()

# --- INCLUDE USED CATEGORIES ---
new_categories = [cat for cat in coco.loadCats(list(used_category_ids))]

# --- INCLUDE TARGET IMAGES (download only) ---
all_target_ids = set(selection["targets"])
all_image_ids = all_target_ids.union(used_image_ids)

for image_id in all_target_ids:
    img_info = coco.loadImgs(image_id)[0]
    img_filename = img_info['file_name']
    img_path = os.path.join(images_dir, img_filename)
    if not os.path.exists(img_path):
        url = coco_images_url.format(img_filename.split('.')[0])
        img_data = requests.get(url).content
        with open(img_path, 'wb') as f:
            f.write(img_data)

# --- SAVE NEW ANNOTATIONS ---
custom_json = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": new_categories
}

with open(os.path.join(annotations_dir, "custom_references.json"), 'w') as f:
    json.dump(custom_json, f)
    
# --- SAVE TARGET IMAGES WITHOUT ANNOTATIONS ---
target_images = [coco.loadImgs(image_id)[0] for image_id in selection["targets"]]

target_annotations = []
for image_id in selection["targets"]:
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns = coco.loadAnns(ann_ids)
    target_annotations.extend(anns)

custom_targets_json = {
    "images": target_images,
    "annotations": target_annotations,
    "categories": new_categories  # Optional, but keeps category info consistent
}

with open(os.path.join(annotations_dir, "custom_targets.json"), 'w') as f:
    json.dump(custom_targets_json, f)

print(f"✅ Saved annotations to {annotations_dir}/custom_references.json")
print(f"✅ Saved target-only metadata to {annotations_dir}/custom_targets.json")   
print(f"✅ Downloaded {len(all_image_ids)} images into {images_dir}")
print(f"✅ Visualisations saved in {vis_dir}")
