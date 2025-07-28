import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator

class IlluminationCorrector:
    def __init__(self, reference_image_path, target_image_path, reference_new_path,
                 target_new_path):
        """
        Initialize with paths to reference (good) and target (worse) images
        """
        self.reference = cv2.imread(reference_image_path)
        self.target = cv2.imread(target_image_path)
        
        if self.reference is None or self.target is None:
            raise ValueError("Could not load one or both images")
        
        self.reference_new = cv2.imread(reference_new_path)
        if self.reference_new is None:
            raise ValueError(f"Could not load new reference image: {reference_new_path}")
        print(f"New reference image loaded: {self.reference_new.shape}")
        
        self.target_new = cv2.imread(target_new_path)
        if self.target_new is None:
            raise ValueError(f"Could not load new target image: {target_new_path}")
        print(f"New target image loaded: {self.target_new.shape}")
        
        self.reference_rgb = cv2.cvtColor(self.reference, cv2.COLOR_BGR2RGB)
        self.target_rgb = cv2.cvtColor(self.target, cv2.COLOR_BGR2RGB)
        
        print(f"Reference image shape: {self.reference.shape}")
        print(f"Target image shape: {self.target.shape}")
        
    def extract_paper_region(self, crop_percentage=0.1):

        h, w = self.reference.shape[:2]
        
        crop_h = int(h * crop_percentage)
        crop_w = int(w * crop_percentage)
        
        ref_paper = self.reference[crop_h:h-crop_h, crop_w:w-crop_w]
        target_paper = self.target[crop_h:h-crop_h, crop_w:w-crop_w]
        
        # cv2.imshow("Reference", cv2.resize(ref_paper, (600, 400)))
        # cv2.imshow("Target", cv2.resize(target_paper, (600, 400)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        #TODO add automatic paper detection, resize both paper images
        # to the same size
        
        return ref_paper, target_paper, (crop_h, crop_w)
    
    def calculate_illumination_ratio_sampled(self, ref_paper, target_paper, 
                                           sample_step=20, region_size=10):

        h, w = ref_paper.shape[:2]
        
        ref_gray = cv2.cvtColor(ref_paper.astype(np.float32), cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_paper.astype(np.float32), cv2.COLOR_BGR2GRAY)
        
        vis_image = target_paper.copy()
        
        half_region = region_size // 2
        sample_points = []
        sample_ratios = []
        
        for y in range(half_region, h - half_region, sample_step):
            for x in range(half_region, w - half_region, sample_step):
                ref_region = ref_gray[y-half_region:y+half_region+1, 
                                    x-half_region:x+half_region+1]
                target_region = target_gray[y-half_region:y+half_region+1, 
                                          x-half_region:x+half_region+1]
                
                ref_avg = np.mean(ref_region)
                target_avg = np.mean(target_region)
                
                if target_avg > 1.0:
                    ratio = ref_avg / target_avg
                    sample_points.append((x, y))
                    sample_ratios.append(ratio)
                    
                cv2.rectangle(vis_image, 
                                (x-half_region, y-half_region), 
                                (x+half_region, y+half_region), 
                                (0, 255, 0), 1)
        
        sample_points = np.array(sample_points)
        sample_ratios = np.array(sample_ratios)
        
        print(f"Created {len(sample_points)} sample points")
        print(f"Ratio range: {sample_ratios.min():.3f} to {sample_ratios.max():.3f}")
        
        # Display the visualization
        plt.figure(figsize=(12, 8))
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        plt.imshow(vis_image_rgb)
        plt.title(f'Sampling Points and Regions\n'
                 f'Sample step: {sample_step}px, Region size: {region_size}x{region_size}px\n'
                 f'Total points: {len(sample_points)}')
        plt.axis('off')
        
        # Add legend
        from matplotlib.patches import Rectangle, Circle
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                   markersize=8, label=f'{region_size}x{region_size} sampling regions'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        return sample_points, sample_ratios

    def interpolate_illumination_map(self, sample_points, sample_ratios, target_shape, interpolation_method):
  
        h, w = target_shape[:2]

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        
        if interpolation_method == 'rbf':
            try:
                rbf = RBFInterpolator(sample_points, sample_ratios, 
                                    kernel='thin_plate_spline', 
                                    smoothing=0.1)
                interpolated = rbf(grid_points)
            except:
                print("RBF failed, falling back to cubic")
                interpolated = griddata(sample_points, sample_ratios, grid_points, 
                                      method='cubic', fill_value=1.0)
        else:
            interpolated = griddata(sample_points, sample_ratios, grid_points, 
                                  method=interpolation_method, fill_value=1.0)
        
        mask = np.isnan(interpolated)
        if np.any(mask):
            print(f"Filling {np.sum(mask)} NaN values with nearest neighbor")
            interpolated_nearest = griddata(sample_points, sample_ratios, grid_points, 
                                          method='nearest')
            interpolated[mask] = interpolated_nearest[mask]
            
        return interpolated.reshape(h, w)
            
    def create_smooth_illumination_map(self, illumination_ratio, sigma=20):

        kernel_size = int(2 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        smooth_map = cv2.GaussianBlur(illumination_ratio, 
                                    (kernel_size, kernel_size), 
                                    sigma)
        
        return smooth_map
    
    def resize_map_to_full_image(self, illumination_map, crop_coords):

        crop_h, crop_w = crop_coords
        full_h, full_w = self.target.shape[:2]
        
        map_resized = cv2.resize(illumination_map, 
                               (full_w - 2*crop_w, full_h - 2*crop_h), 
                               interpolation=cv2.INTER_CUBIC)
        
        full_map = np.ones((full_h, full_w), dtype=np.float32)
        
        full_map[crop_h:full_h-crop_h, crop_w:full_w-crop_w] = map_resized
        
        full_map[:crop_h, crop_w:full_w-crop_w] = map_resized[0:1, :]
        full_map[full_h-crop_h:, crop_w:full_w-crop_w] = map_resized[-1:, :]

        full_map[:, :crop_w] = full_map[:, crop_w:crop_w+1]
        full_map[:, full_w-crop_w:] = full_map[:, full_w-crop_w-1:full_w-crop_w]
        
        return full_map
    
    def apply_correction_to_target_new(self, illumination_map, target_new=None):

        if target_new is None:
            if self.target_new is None:
                raise ValueError("No new image provided")
            target_new = self.target_new
        
        print("Applying illumination correction to new image...")
        corrected = target_new.astype(np.float32)
        
        for channel in range(3):
            corrected[:, :, channel] *= illumination_map
        
        corrected = np.clip(corrected, 0, 255)
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        return corrected_rgb.astype(np.uint8)
    
    
    def apply_kries_color_adaptation(self, src_img, src_white_rgb, dst_white_rgb):

        RGB_to_LMS = np.array([
            [0.4002, 0.7076, -0.0808],
            [-0.2263, 1.1653, 0.0457],
            [0.0000, 0.0000, 0.9182]
        ])

        LMS_to_RGB = np.linalg.inv(RGB_to_LMS)

        src_white_LMS = RGB_to_LMS @ src_white_rgb
        dst_white_LMS = RGB_to_LMS @ dst_white_rgb

        D = np.diag(dst_white_LMS / (src_white_LMS + 1e-6))

        adaptation_matrix = LMS_to_RGB @ D @ RGB_to_LMS

        img_float = src_img.astype(np.float32).copy() / 255
        reshaped = img_float.reshape(-1, 3).T
        adapted = adaptation_matrix @ reshaped
        adapted = np.clip(adapted.T.reshape(src_img.shape), 0, 1)
        adapted_img = (adapted * 255).astype(np.uint8)
        
        return adapted_img

    
    def process(self, crop_percentage=0.1, smoothing_sigma=20,
                sample_step=20, region_size=10, interpolation_method='cubic'):

        print("Extracting paper regions...")
        ref_paper, target_paper, crop_coords = self.extract_paper_region(crop_percentage)
        
        ref_white_rgb = np.mean(ref_paper.reshape(-1, 3), axis=0)
        target_white_rgb = np.mean(target_paper.reshape(-1, 3), axis=0)
        
        print("Calculating illumination ratios from sampled points...")
        sample_points, sample_ratios = self.calculate_illumination_ratio_sampled(
            ref_paper, target_paper, sample_step, region_size)
        
        print("Creating illumination map through interpolation...")
        illumination_map = self.interpolate_illumination_map(
            sample_points, sample_ratios, ref_paper.shape, interpolation_method)
        
        print("Smoothing illumination map...")
        smooth_map = self.create_smooth_illumination_map(illumination_map, smoothing_sigma)
        
        print("Resizing map to full image...")
        full_illumination_map = self.resize_map_to_full_image(smooth_map, crop_coords)
        
        print("Applying to the new image...")
        corrected_target_new = self.apply_correction_to_target_new(full_illumination_map)
        
        print("Applying Kries color adaptation...")
        kries_target_new= self.apply_kries_color_adaptation(
            corrected_target_new, target_white_rgb, ref_white_rgb
        )
        
        corrected_rgb = cv2.cvtColor(corrected_target_new, cv2.COLOR_BGR2RGB)
        kries_rgb     = cv2.cvtColor(kries_target_new, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(14, 7))

        plt.subplot(1, 2, 1)
        plt.imshow(corrected_rgb)
        plt.title("After Illumination Correction")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(kries_rgb)
        plt.title("After Kries Color Correction")
        plt.axis('off')

        plt.tight_layout()
        # plt.show()
        
        return corrected_target_new, full_illumination_map
    
    def visualize_results(self, corrected_image, illumination_map, save_path):

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
                
        # Reference image
        axes[0, 0].imshow(self.reference_rgb)
        axes[0, 0].set_title('Reference Image (Good Conditions)')
        axes[0, 0].axis('off')
        
        # Original target image
        axes[0, 1].imshow(self.target_rgb)
        axes[0, 1].set_title('Original Target Image (Worse Conditions)')
        axes[0, 1].axis('off')
    
        # Illumination map
        axes[0, 2].imshow(illumination_map, cmap='jet')
        axes[0, 2].set_title('Illumination Map')
        axes[0, 2].axis('off')
        
        # New reference image
        axes[1, 0].imshow(self.reference_new)
        axes[1, 0].set_title('New referece image')
        axes[1, 0].axis('off')
        
        # New target image
        axes[1, 1].imshow(self.target_new)
        axes[1, 1].set_title('Actual target image')
        axes[1, 1].axis('off')
        
        # Corrected new target image
        corrected_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
        axes[1, 2].imshow(corrected_rgb)
        axes[1, 2].set_title('Corrected Image')
        axes[1, 2].axis('off')
        
        plt.subplots_adjust(hspace=4, wspace=2)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        

# Usage example
if __name__ == "__main__":
    
    reference_con = '0_ls8'
    target_con = '4_ls4'
    corrections_dir = '/home/vicosdemo/Documents/dataset/work_ploscice/corrections'

    corrector = IlluminationCorrector(
        reference_image_path=f"/home/vicosdemo/Documents/dataset/work_ploscice/white/{reference_con}.jpg",
        target_image_path=f"/home/vicosdemo/Documents/dataset/work_ploscice/white/{target_con}.jpg",
        reference_new_path=f"/home/vicosdemo/Documents/dataset/work_ploscice/tile images/{reference_con}.jpg",
        target_new_path = f"/home/vicosdemo/Documents/dataset/work_ploscice/tile images/{target_con}.jpg"
    )
    
    # Process the images
    corrected_image, illumination_map = corrector.process(
        crop_percentage=0.2,
        smoothing_sigma=301,
        sample_step=50,
        region_size=10,
        interpolation_method='cubic'
    )
    
    # Visualize results
    visualiation = corrector.visualize_results(corrected_image, illumination_map, corrections_dir + f'/{reference_con}-{target_con}.jpg')

    # Save the corrected image
    # cv2.imwrite("corrected_image.jpg", corrected_image)
    
    # Save the illumination map for analysis
    map_normalized = ((illumination_map - illumination_map.min()) / 
                     (illumination_map.max() - illumination_map.min()) * 255).astype(np.uint8)
    # cv2.imwrite("illumination_map.jpg", map_normalized)
    
    print("Processing complete!")
    print(f"Illumination map range: {illumination_map.min():.3f} to {illumination_map.max():.3f}")
    # print("Files saved: corrected_image.jpg, illumination_map.jpg")