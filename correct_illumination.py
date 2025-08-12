import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator

class IlluminationCorrector:
    def __init__(self, reference_image_path, target_image_path, reference_new_path,
            target_new_path, reference_cal_path, target_cal_path):

        self.reference = cv2.imread(reference_image_path)
        self.target = cv2.imread(target_image_path)
        
        self.reference_new = cv2.imread(reference_new_path)
        self.target_new = cv2.imread(target_new_path)
        self.reference_rgb = cv2.cvtColor(self.reference, cv2.COLOR_BGR2RGB)
        self.target_rgb = cv2.cvtColor(self.target, cv2.COLOR_BGR2RGB)
        
        self.reference_cal = cv2.imread(reference_cal_path)
        self.target_cal = cv2.imread(target_cal_path)
        self.reference_cal_rgb = cv2.cvtColor(self.reference_cal, cv2.COLOR_BGR2RGB)
        self.target_cal_rgb = cv2.cvtColor(self.target_cal, cv2.COLOR_BGR2RGB)
        
        if self.reference is None or self.target is None or \
            self.reference_new is None or self.target_new is None or \
            self.reference_cal is None or self.target_cal is None:
            raise ValueError("One or more image paths are invalid or images could not be loaded.")

        
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
        
        # print(f"Extracted paper region: {ref_paper.shape} for reference and {target_paper.shape} for target")
        # print(f"White ROI coordinates: {crop_h}, {crop_w}")
        
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
                
                # print(ref_region.size, target_region.size)
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
        print(f"Sample points: {sample_points[:5]}")  # Show first 5 points for debugging
        
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
    
    def illumination_correction(self, illumination_map, target_new=None):

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

    # def match_histograms(self, source, reference):

    #     print("Applying histogram matching...")
        
    #     matched = np.zeros_like(source)
        
    #     for channel in range(3):
    #         # Get histograms
    #         source_hist, source_bins = np.histogram(source[:, :, channel].flatten(), 
    #                                             bins=256, range=(0, 256))
    #         ref_hist, ref_bins = np.histogram(reference[:, :, channel].flatten(), 
    #                                         bins=256, range=(0, 256))
            
    #         # Calculate CDFs (Cumulative Distribution Functions)
    #         source_cdf = np.cumsum(source_hist).astype(np.float32)
    #         ref_cdf = np.cumsum(ref_hist).astype(np.float32)
            
    #         # Normalize CDFs
    #         source_cdf = source_cdf / source_cdf[-1]
    #         ref_cdf = ref_cdf / ref_cdf[-1]
            
    #         # Create lookup table
    #         lookup_table = np.zeros(256, dtype=np.uint8)
            
    #         for i in range(256):
    #             # Find the closest CDF value in reference
    #             closest_idx = np.argmin(np.abs(ref_cdf - source_cdf[i]))
    #             lookup_table[i] = closest_idx
            
    #         # Apply lookup table
    #         matched[:, :, channel] = lookup_table[source[:, :, channel]]
        
    #     return matched
    
    def compute_color_correction_matrix(self, source_rgb, target_rgb):
        """
        source_rgb: (N, 3) target calibration image pixels (after illumination correction)
        target_rgb: (N, 3) reference calibration image pixels
        Returns a 3x3 matrix to convert source → reference
        """
        A = source_rgb.reshape(-1, 3).astype(np.float32)
        B = target_rgb.reshape(-1, 3).astype(np.float32)
        M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        return M.T
    
    def apply_color_correction_matrix(self, image_rgb, matrix):
        h, w, _ = image_rgb.shape
        reshaped = image_rgb.reshape(-1, 3).astype(np.float32)
        corrected = reshaped @ matrix.T
        corrected = np.clip(corrected, 0, 255)
        return corrected.reshape(h, w, 3).astype(np.uint8)
    
    def compute_histogram_luts(self, source_rgb, reference_rgb):
        """
        Computes per-channel histogram LUTs to match source → reference.
        Both images must be in RGB format.

        Returns:
            luts: List of 3 LUTs (one for each channel)
        """
        def get_lut(src_channel, ref_channel):
            src_hist, _ = np.histogram(src_channel.flatten(), bins=256, range=(0, 256))
            ref_hist, _ = np.histogram(ref_channel.flatten(), bins=256, range=(0, 256))

            src_cdf = np.cumsum(src_hist).astype(np.float32)
            ref_cdf = np.cumsum(ref_hist).astype(np.float32)
            src_cdf /= src_cdf[-1]
            ref_cdf /= ref_cdf[-1]

            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                idx = np.argmin(np.abs(ref_cdf - src_cdf[i]))
                lut[i] = idx
            return lut

        luts = [get_lut(source_rgb[:, :, c], reference_rgb[:, :, c]) for c in range(3)]
        return luts
    
    def compute_histogram_luts_searchsorted(self, source_rgb, reference_rgb):
        """
        Alternative version using np.searchsorted instead of argmin.
        """
        def calculate_cdfs(image):
            channels = cv2.split(image)
            cdfs = []
            for channel in channels:
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                cdf = hist.cumsum()
                cdfs.append(cdf / cdf.max())
            return cdfs

        src_cdfs = calculate_cdfs(source_rgb)
        ref_cdfs = calculate_cdfs(reference_rgb)
        
        luts = []
        for i in range(3):
            lut = np.zeros(256, dtype=np.uint8)
            for j in range(256):
                lut[j] = np.searchsorted(ref_cdfs[i], src_cdfs[i][j])
            luts.append(lut)
        
        return luts
        
    def apply_histogram_luts(self, image_rgb, luts):
        """
        Applies per-channel LUTs to an RGB image.

        Args:
            image_rgb: Input RGB image.
            luts: List of 3 LUTs (R, G, B)

        Returns:
            RGB image after applying LUTs.
        """
        corrected_channels = [cv2.LUT(image_rgb[:, :, c], luts[c]) for c in range(3)]
        return cv2.merge(corrected_channels)
    
    def visualize_calibration_correction(self, corrected_cal_target, save_path=None):
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Reference calibration
        axes[0].imshow(self.reference_cal_rgb)
        axes[0].set_title('Reference Calibration\n(Good Conditions)')
        axes[0].axis('off')
        
        # Target calibration (original)
        axes[1].imshow(self.target_cal_rgb)
        axes[1].set_title('Target Calibration\n(Worse Conditions - Original)')
        axes[1].axis('off')
        
        # Corrected target calibration
        axes[2].imshow(corrected_cal_target)
        axes[2].set_title('Target Calibration\n(Illumination Corrected)')
        axes[2].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path.replace('.jpg', '_calibration.jpg'), dpi=300, bbox_inches='tight')
        plt.show()
        
            
    def process(self, crop_percentage=0.1, smoothing_sigma=20,
                sample_step=20, region_size=10, interpolation_method='cubic'):

        print("=== STEP 1: ILLUMINATION CORRECTION ===")
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
        
        plt.imshow(illumination_map, cmap='viridis')
        plt.title("Illumination Map")
        plt.legend(['Illumination Ratio'])
        plt.colorbar(label='Relative Illumination')
        plt.tight_layout()
        plt.axis('off')
        plt.show()
        print("Smoothing illumination map...")
        smooth_map = self.create_smooth_illumination_map(illumination_map, smoothing_sigma)
        
        print("Resizing map to full image...")
        full_illumination_map = self.resize_map_to_full_image(smooth_map, crop_coords)
        
        # Apply illumination correction to calibration target
        print("Applying illumination correction to calibration target...")
        corrected_cal_target = self.illumination_correction(full_illumination_map, self.target_cal)
        
        # Visualize calibration correction
        print("Displaying calibration correction results...")
        self.visualize_calibration_correction(corrected_cal_target)
        
        print("Applying illumination correction to new target image...")
        corrected_new_target = self.illumination_correction(full_illumination_map, self.target_new)
        
        
        print("\n=== STEP 2: COLOR CORRECTION ===")
        
        self.visualize_patch_grid_on_image(
            corrected_cal_target, 
            (190, 420),
            (1740, 1200),
            grid_shape=(4, 7),
            sample_size=50
        )
        self.visualize_patch_grid_on_image(
            self.reference_cal_rgb, 
            (190, 420),
            (1740, 1200),
            grid_shape=(4, 7),
            sample_size=50
        )
        
        
        M = self.compute_ccm_from_patch_grid(
            corrected_cal_target,
            self.reference_cal_rgb,
            (190, 420),
            (1740, 1200),
            grid_shape=(4, 7),
            sample_size=100
        )
        
        M_applied = self.apply_ccm(
            corrected_new_target,
            M
        )
        
        print(M)
        
        display_size = (800, 600)
        final_corrected_resized = cv2.resize(M_applied, display_size)
        
        # ccm = self.compute_color_correction_matrix(corrected_cal_target, self.reference_cal_rgb)
        # final_corrected_target = self.apply_color_correction_matrix(corrected_new_target, ccm)
        
        # luts = self.compute_histogram_luts(corrected_new_target, self.reference_new)
        # luts = self.compute_histogram_luts_searchsorted(corrected_new_target, self.reference_new)
        # final_corrected_target = self.apply_histogram_luts(corrected_new_target, luts)
        
        # Resize images for display
        # display_size = (800, 600)
        # final_corrected_resized = cv2.resize(final_corrected_target, display_size)
        target_new_resized = cv2.resize(self.target_new, display_size)
        reference_new_resized = cv2.resize(self.reference_new, display_size)
        
        cv2.imshow("Final Corrected Target1", cv2.cvtColor(final_corrected_resized, cv2.COLOR_BGR2RGB))
        cv2.imshow("Original Target1", target_new_resized)
        cv2.imshow("Original Reference1", reference_new_resized)

        # cv2.imshow("Final Corrected Target", cv2.cvtColor(final_corrected_resized, cv2.COLOR_BGR2RGB))
        # cv2.imshow("Original Target", cv2.cvtColor(target_new_resized, cv2.COLOR_BGR2RGB))
        # cv2.imshow("Original Reference", cv2.cvtColor(reference_new_resized, cv2.COLOR_BGR2RGB))


        
        
        # print("\n=== STEP 2: HISTOGRAM MATCHING ===")
        # luts = self.compute_histogram_luts(corrected_cal_target, self.reference_cal)
        # # luts = self.compute_histogram_luts_searchsorted(corrected_new_target, self.reference_new)
        # final_matched = self.apply_histogram_luts(M_applied, luts)
        
        # cv2.imshow("Final Matched Image", cv2.cvtColor(final_matched, cv2.COLOR_BGR2RGB))
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # # Apply histogram matching using corrected calibration images
        # print("Matching histograms between corrected calibration images...")
        # histogram_matched_cal = self.match_histograms(corrected_cal_target, self.reference_cal)
        
        # # Apply the same histogram transformation to the new image
        # print("Applying histogram matching to new target image...")
        # final_corrected_new = self.match_histograms(corrected_target_new, self.reference_new)
        
        # # # Convert final result to RGB for visualization
        # # final_corrected_new_rgb = cv2.cvtColor(final_corrected_new, cv2.COLOR_BGR2RGB)
        
        
        return corrected_new_target, full_illumination_map
    
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
        
    
    def compute_ccm_from_patch_grid(self, target_rgb, reference_rgb, top_left, bottom_right, grid_shape=(4, 7), sample_size=10):

        rows, cols = grid_shape
        patch_width = (bottom_right[0] - top_left[0]) / cols
        patch_height = (bottom_right[1] - top_left[1]) / rows

        sampled_colors_target = []
        sampled_colors_reference = []

        for row in range(rows):
            for col in range(cols):
                # Center of the current patch
                center_x = int(top_left[0] + (col + 0.5) * patch_width)
                center_y = int(top_left[1] + (row + 0.5) * patch_height)

                # Define sampling region
                half_size = sample_size // 2
                x_start = max(center_x - half_size, 0)
                x_end = min(center_x + half_size + 1, target_rgb.shape[1])
                y_start = max(center_y - half_size, 0)
                y_end = min(center_y + half_size + 1, target_rgb.shape[0])

                target_patch = target_rgb[y_start:y_end, x_start:x_end]
                reference_patch = reference_rgb[y_start:y_end, x_start:x_end]
                # mean_color = np.mean(patch.reshape(-1, 3), axis=0)
                # sampled_colors_target.append(mean_color)
                
                sampled_colors_target.append(np.mean(target_patch.reshape(-1, 3), axis=0))
                sampled_colors_reference.append(np.mean(reference_patch.reshape(-1, 3), axis=0))

        # Solve for CCM: reference ≈ M * sampled → M = least squares solution
        A = np.array(sampled_colors_target, dtype=np.float32)
        B = np.array(sampled_colors_reference, dtype=np.float32)
        print("A shape (target):", A.shape)
        print("B shape (reference):", B.shape)

        M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

        return M.T
    
    def visualize_patch_grid_on_image(self, image_rgb, top_left, bottom_right, grid_shape=(4, 7), sample_size=10):

        vis_img = image_rgb.copy()
        rows, cols = grid_shape
        patch_width = (bottom_right[0] - top_left[0]) / cols
        patch_height = (bottom_right[1] - top_left[1]) / rows
        half_size = sample_size // 2

        for row in range(rows):
            for col in range(cols):
                center_x = int(top_left[0] + (col + 0.5) * patch_width)
                center_y = int(top_left[1] + (row + 0.5) * patch_height)
                cv2.rectangle(vis_img,
                            (center_x - half_size, center_y - half_size),
                            (center_x + half_size, center_y + half_size),
                            (255, 0, 0), 1)
                cv2.circle(vis_img, (center_x, center_y), 2, (0, 255, 0), -1)

        plt.figure(figsize=(10, 6))
        plt.imshow(vis_img)
        plt.title("Sample Regions for Calibration Patches")
        plt.axis('off')
        plt.show()
    
    def apply_ccm(self, image_rgb, ccm_matrix):
        """
        Applies a 3x3 CCM to an RGB image.

        Args:
            image_rgb (np.ndarray): Input RGB image.
            ccm_matrix (np.ndarray): 3x3 color correction matrix.

        Returns:
            np.ndarray: Color-corrected RGB image.
        """
        h, w, _ = image_rgb.shape
        reshaped = image_rgb.reshape(-1, 3).astype(np.float32)
        corrected = reshaped @ ccm_matrix.T
        corrected = np.clip(corrected, 0, 255)
        return corrected.reshape(h, w, 3).astype(np.uint8)
        

# Usage example
if __name__ == "__main__":
    
    reference_con = '0_ls8'
    target_con = '3_ls3'
    white_cal_dir = './white'
    tile_images_dir = './tile images'
    cal_dir = './cal'
    corrections_dir = './corrections'

    corrector = IlluminationCorrector(
        reference_image_path=f"{white_cal_dir}/{reference_con}.jpg",
        target_image_path=f"{white_cal_dir}/{target_con}.jpg",
        reference_new_path=f"{tile_images_dir}/{reference_con}.jpg",
        target_new_path = f"{tile_images_dir}/{target_con}.jpg",
        reference_cal_path = f"{cal_dir}/{reference_con}.jpg",
        target_cal_path = f"{cal_dir}/{target_con}.jpg"
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