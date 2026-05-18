from PIL import Image
import os

def concat_horizontally(im1, im2):
    # Resize im2 to match im1 height if needed
    if im1.height != im2.height:
        new_width = int(im2.width * (im1.height / im2.height))
        im2 = im2.resize((new_width, im1.height), Image.LANCZOS)
    
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def concat_vertically(im1, im2):
    # Resize im2 to match im1 width if needed
    if im1.width != im2.width:
        new_height = int(im2.height * (im1.width / im2.width))
        im2 = im2.resize((im1.width, new_height), Image.LANCZOS)
    
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def combine_features(features, output_name):
    rows = []
    print(f"Creating {output_name}...")
    for map_path, plot_path in features:
        if not os.path.exists(map_path) or not os.path.exists(plot_path):
            print(f"Missing file! Map: {map_path}, Plot: {plot_path}")
            continue
        im_map = Image.open(map_path).convert('RGB')
        im_plot = Image.open(plot_path).convert('RGB')
        
        # Combine horizontally into a row
        row = concat_horizontally(im_map, im_plot)
        rows.append(row)
    
    if len(rows) == 2:
        # Combine rows vertically
        fig = concat_vertically(rows[0], rows[1])
        fig.save(output_name)
        print(f"Saved: {output_name}")
    else:
        print(f"Failed to create {output_name}, missing pieces.")

def main():
    base_dir = "."
    maps_dir = os.path.join(base_dir, "Spatial_Maps")
    plots_dir = os.path.join(base_dir, "Dependence_Plots_LOESS")
    
    # Feature names
    f1 = "Human_Dev_Index_Local"
    f2 = "Cropland_Extent_Local"
    f3 = "Potential_Evap_Local"
    f4 = "Human_Footprint_Upstream"
    
    # Figure 1: HD and Crop
    fig1_features = [
        (os.path.join(maps_dir, f"Map_SHAP_{f1}.png"), os.path.join(plots_dir, f"Dependence_{f1}.png")),
        (os.path.join(maps_dir, f"Map_SHAP_{f2}.png"), os.path.join(plots_dir, f"Dependence_{f2}.png"))
    ]
    
    # Figure 2: PET and HFi
    fig2_features = [
        (os.path.join(maps_dir, f"Map_SHAP_{f3}.png"), os.path.join(plots_dir, f"Dependence_{f3}.png")),
        (os.path.join(maps_dir, f"Map_SHAP_{f4}.png"), os.path.join(plots_dir, f"Dependence_{f4}.png"))
    ]
    
    combine_features(fig1_features, "Top4_Feature_Combined_1.png")
    combine_features(fig2_features, "Top4_Feature_Combined_2.png")

if __name__ == "__main__":
    main()
