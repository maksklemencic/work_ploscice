import cv2
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

class BaseCorrector:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.images = pipeline.get_images()

class IlluminationCorrector(BaseCorrector):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        
        self.ref_img = self.images['reference_tile']
        self.tar_img = self.images['target_tile']
        self.ref_white = self.images['reference_white']
        self.tar_white = self.images['target_white']
        self.ref_cal = self.images['reference_calibration']
        self.tar_cal = self.images['target_calibration']
        
        self.settings = pipeline.illumination_settings
        
    def extract_paper_region(self):
        
        def order_points(pts):
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]
                return rect

        def crop_with_quad(image, quad_pts):
            rect = order_points(quad_pts)
            (tl, tr, br, bl) = rect

            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped
    
        def get_crop_margins(corners, full_image_shape):
            full_h, full_w = full_image_shape[:2]

            x_min = np.min(corners[:, 0])
            x_max = np.max(corners[:, 0])
            y_min = np.min(corners[:, 1])
            y_max = np.max(corners[:, 1])

            crop_left   = int(round(x_min))
            crop_right  = int(round(full_w - x_max))
            crop_top    = int(round(y_min))
            crop_bottom = int(round(full_h - y_max))

            return crop_top, crop_bottom, crop_left, crop_right

        ref_corners, ref_area = self.pipeline.detect_sample_region(self.ref_white)
        if ref_corners is None or len(ref_corners) != 4:
            raise ValueError("Could not detect paper region in reference image.")

        tar_corners, _ = self.pipeline.detect_sample_region(self.tar_white, reference_area=ref_area)
        if tar_corners is None or len(tar_corners) != 4:
            raise ValueError("Could not detect paper region in target image.")

        ref_paper = crop_with_quad(self.ref_white, ref_corners)
        target_paper = crop_with_quad(self.tar_white, tar_corners)
        
        if self.settings.get("visualize", True):
            self.pipeline.preview_sample_region(ref_corners, self.ref_white, ref_paper, show=False)
            self.pipeline.preview_sample_region(tar_corners, self.tar_white, target_paper)
        
        ref_margins = get_crop_margins(ref_corners, self.ref_white.shape)
        tar_margins = get_crop_margins(tar_corners, self.tar_white.shape)
        
        ref_h, ref_w = ref_paper.shape[:2]
        tar_h, tar_w = target_paper.shape[:2]

        final_h = min(ref_h, tar_h)
        final_w = min(ref_w, tar_w)

        ref_paper_resized = cv2.resize(ref_paper, (final_w, final_h), interpolation=cv2.INTER_AREA)
        target_paper_resized = cv2.resize(target_paper, (final_w, final_h), interpolation=cv2.INTER_AREA)

        crop_top    = max(ref_margins[0], tar_margins[0])
        crop_bottom = max(ref_margins[1], tar_margins[1])
        crop_left   = max(ref_margins[2], tar_margins[2])
        crop_right  = max(ref_margins[3], tar_margins[3])
        
        self.crop_coords = (crop_top, crop_bottom, crop_left, crop_right)

        return ref_paper_resized, target_paper_resized
    
    def calculate_illumination_ratios(self, ref_white_roi, tar_white_roi):
        h, w = ref_white_roi.shape[:2]
        region_size = self.settings.get("region_size", 10)
        sample_step = self.settings.get("sample_step", 30)
        
        ref_gray = cv2.cvtColor(ref_white_roi.astype(np.float32), cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(tar_white_roi.astype(np.float32), cv2.COLOR_BGR2GRAY)
        
        half_region = region_size // 2
        sample_points = []
        sample_ratios = []
        
        if self.settings.get("visualize", True):
            vis_image = tar_white_roi.copy()            
        
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
                    
                if self.settings.get("visualize", True):
                    cv2.rectangle(vis_image, 
                                (x - half_region, y - half_region), 
                                (x + half_region, y + half_region), 
                                (0, 255, 0), 1)
        
        sample_points = np.array(sample_points)
        sample_ratios = np.array(sample_ratios)
        
        if self.settings.get("visualize", True):
            self.visualize_paper_regions(vis_image, sample_points, region_size, sample_step)
        
        return sample_points, sample_ratios

    def interpolate_illumination_map(self, sample_points, sample_ratios, white_shape):
        h, w = white_shape[:2]
        
        interpolation_method = self.settings.get("interpolation_method", "cubic")
        supported_methods = ['rbf', 'linear', 'cubic', 'nearest']
        
        if interpolation_method not in supported_methods:
            raise ValueError(f"Unsupported interpolation method: {interpolation_method}. "
                            f"Supported methods are: {supported_methods}")

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        grid_points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        
        fill_val = np.median(sample_ratios)
        
        if interpolation_method == 'rbf':
            rbf = RBFInterpolator(
                sample_points,
                sample_ratios,
                kernel='thin_plate_spline',
                smoothing=0.1
            )
            interpolated = rbf(grid_points)
        else:
            interpolated = griddata(
                sample_points,
                sample_ratios,
                grid_points,
                method=interpolation_method,
                fill_value=fill_val
            )
            
        if np.any(np.isnan(interpolated)):
            interpolated_nearest = griddata(
                sample_points,
                sample_ratios,
                grid_points,
                method='nearest'
            )
            interpolated[np.isnan(interpolated)] = interpolated_nearest[np.isnan(interpolated)]

        return interpolated.reshape(h, w)

    def smooth_map(self, map):
        sigma = self.settings.get("smoothing_sigma", 301)
        kernel_size = int(2 * sigma + 1)
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        smooth_map = cv2.GaussianBlur(map, 
                                    (kernel_size, kernel_size), 
                                    sigma)
        
        return smooth_map
    
    def resize_map_to_full(self, map):
        crop_top, crop_bottom, crop_left, crop_right = self.crop_coords
        full_h, full_w = self.tar_img.shape[:2]

        target_h = full_h - crop_top - crop_bottom
        target_w = full_w - crop_left - crop_right

        map_resized = cv2.resize(map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        full_map = np.ones((full_h, full_w), dtype=np.float32)

        full_map[crop_top:full_h - crop_bottom, crop_left:full_w - crop_right] = map_resized
        full_map[:crop_top, crop_left:full_w - crop_right] = map_resized[0:1, :]               # top
        full_map[full_h - crop_bottom:, crop_left:full_w - crop_right] = map_resized[-1:, :]   # bottom

        full_map[:, :crop_left] = full_map[:, crop_left:crop_left + 1]                         # left
        full_map[:, full_w - crop_right:] = full_map[:, full_w - crop_right - 1:full_w - crop_right]  # right

        return full_map

    def apply_map(self, target):
        corrected = target.astype(np.float32)
        for channel in range(3):
            corrected[:, :, channel] *= self.illumination_map
        
        corrected = np.clip(corrected, 0, 255)
        corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        return corrected_rgb.astype(np.uint8)
        
    def visualize(self):
        self.pipeline.visualize_3images(
            [self.ref_cal, self.tar_cal, self.tar_cal_illumination],
            ["Reference Calibration", "Target Calibration", "Illumination Corrected Target Calibration"],
            show=False
        )
        self.pipeline.visualize_3images(
            [self.ref_img, self.tar_img, self.tar_img_illumination],
            ["Reference Image", "Target Image", "Illumination Corrected Target Image"],
            show=False
        )
        
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        im = ax.imshow(self.illumination_map, cmap='viridis')
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative Illumination")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        map_rgb = np.array(Image.open(buf).convert('RGB'))
        
        self.pipeline.visualize_3images(
            [self.ref_white, self.tar_white, map_rgb],
            ["Reference White", "Target White", "Illumination Map"],
            resize_to=(900, 600)
        )    
    
    def visualize_paper_regions(self, vis_image, sample_points, region_size, sample_step):
        plt.figure(figsize=(12, 8))
        
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        plt.imshow(vis_image_rgb)
        plt.title(f'Sampling Points and Regions\n'
                f'Sample step: {sample_step}px, Region size: {region_size}x{region_size}px\n'
                f'Total points: {len(sample_points)}')
        plt.axis('off')
        
        from matplotlib.patches import Rectangle, Circle
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                markersize=8, label=f'{region_size}x{region_size} sampling regions'),
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
    def process(self):
        ref_white_roi, tar_white_roi = self.extract_paper_region()
        sample_points, sample_ratios = self.calculate_illumination_ratios(ref_white_roi, tar_white_roi)

        print("Interpolating illumination map...")
        illumination_map = self.interpolate_illumination_map(sample_points, sample_ratios, ref_white_roi.shape)
        smoothed_map = self.smooth_map(illumination_map)
        self.illumination_map = self.resize_map_to_full(smoothed_map)
        
        self.tar_cal_illumination = cv2.cvtColor(self.apply_map(self.tar_cal), cv2.COLOR_BGR2RGB)
        self.tar_img_illumination = cv2.cvtColor(self.apply_map(self.tar_img), cv2.COLOR_BGR2RGB)
        
        if self.settings.get("visualize", True):
            self.visualize()
            
        return self.tar_cal_illumination, self.tar_img_illumination
        

class ColorCorrector(BaseCorrector):
    def __init__(self, pipeline):
        super().__init__(pipeline)
        
        self.ref_img = self.images['reference_tile']
        self.tar_img = pipeline.tar_img_illumination
        self.ref_cal = self.images['reference_calibration']
        self.tar_cal = pipeline.tar_cal_illumination
        
        self.settings = pipeline.color_settings
        
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect     
        
    def crop_with_quad(self, image, quad_pts):
            rect = self.order_points(quad_pts)
            (tl, tr, br, bl) = rect

            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = int(max(widthA, widthB))

            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = int(max(heightA, heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped

    def segment_cal_sample(self):
        
        ref_corners = self.pipeline.detect_sample_region_cal(self.ref_cal)
        ref_cal_seg = self.crop_with_quad(self.ref_cal, ref_corners)
        tar_corners = self.pipeline.detect_sample_region_cal(self.tar_cal)
        tar_cal_seg = self.crop_with_quad(self.tar_cal, tar_corners)
        
        ref_h, ref_w = ref_cal_seg.shape[:2]
        tar_h, tar_w = tar_cal_seg.shape[:2]

        final_h = min(ref_h, tar_h)
        final_w = min(ref_w, tar_w)

        self.ref_cal_segment = cv2.resize(ref_cal_seg, (final_w, final_h), interpolation=cv2.INTER_AREA)
        self.tar_cal_segment = cv2.resize(tar_cal_seg, (final_w, final_h), interpolation=cv2.INTER_AREA)

        if self.settings.get("visualize", True):
            self.pipeline.preview_sample_region(ref_corners, self.ref_cal, ref_cal_seg, show=False)
            self.pipeline.preview_sample_region(tar_corners, self.tar_cal, tar_cal_seg)

    def compute_ccm_from_patch_grid_from_segments(self):
        rows, cols = self.settings.get("grid_shape", (4, 7))
        sample_size = self.settings.get("sample_size", 10)

        h, w = self.ref_cal_segment.shape[:2]

        patch_width = w / cols
        patch_height = h / rows

        sampled_colors_target = []
        sampled_colors_reference = []

        for row in range(rows):
            for col in range(cols):
                center_x = int((col + 0.5) * patch_width)
                center_y = int((row + 0.5) * patch_height)

                half_size = sample_size // 2
                x_start = max(center_x - half_size, 0)
                x_end = min(center_x + half_size + 1, w)
                y_start = max(center_y - half_size, 0)
                y_end = min(center_y + half_size + 1, h)

                ref_patch = self.ref_cal_segment[y_start:y_end, x_start:x_end]
                tar_patch = self.tar_cal_segment[y_start:y_end, x_start:x_end]

                ref_mean = np.mean(ref_patch.reshape(-1, 3), axis=0)
                tar_mean = np.mean(tar_patch.reshape(-1, 3), axis=0)

                sampled_colors_reference.append(ref_mean)
                sampled_colors_target.append(tar_mean)

        A = np.array(sampled_colors_target, dtype=np.float32)
        B = np.array(sampled_colors_reference, dtype=np.float32)

        if A.shape != B.shape or A.shape[0] == 0:
            raise ValueError(f"Insufficient or mismatched samples: A={A.shape}, B={B.shape}")

        M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        
        self.ccm = M.T
    
    def apply_ccm(self, image_rgb):
        h, w, _ = image_rgb.shape
        reshaped = image_rgb.reshape(-1, 3).astype(np.float32)
        corrected = reshaped @ self.ccm.T
        corrected = np.clip(corrected, 0, 255)
        return corrected.reshape(h, w, 3).astype(np.uint8)

    def build_histogram_lut(self, source_channel, reference_channel):

        src_hist, _ = np.histogram(source_channel.ravel(), bins=256, range=(0, 256), density=True)
        ref_hist, _ = np.histogram(reference_channel.ravel(), bins=256, range=(0, 256), density=True)

        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)

        lut = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_idx in range(256):
            while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
                ref_idx += 1
            lut[src_idx] = ref_idx

        return lut
    
    def match_histogram_with_luts(self, source_img, reference_img):

        matched = np.zeros_like(source_img)
        luts = []

        for i in range(3):
            lut = self.build_histogram_lut(source_img[:, :, i], reference_img[:, :, i])
            matched[:, :, i] = cv2.LUT(source_img[:, :, i], lut)
            luts.append(lut)

        return matched, luts
    
    def apply_luts_rgb(self, image, luts):
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = cv2.LUT(image[:, :, i], luts[i])
        return result

    def process(self):
        self.segment_cal_sample()
        
        self.compute_ccm_from_patch_grid_from_segments()
        applied_ccm = self.apply_ccm(self.tar_img)
        
        # _, luts = self.match_histogram_with_luts(self.ref_cal_segment, self.tar_cal_segment)
        # matched_image = self.apply_luts_rgb(applied_ccm, luts)
        
        # cv2.imshow("Color Corrected Target Image", matched_image)
        # cv2.imshow("Original Target Image", self.tar_img)
        # cv2.imshow("Reference Image", self.ref_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        if self.settings.get("visualize", True):
            display_size = (800, 600)
            self.pipeline.visualize_3images(
                [self.ref_img, self.tar_img, applied_ccm],
                ["Reference Image", "Target img - Corrected illum", "Color Corrected Target"],
                resize_to=display_size
            )
        

class ImageCorrectionPipeline:
    def __init__(self, reference_paths, target_paths, illumination_settings, color_settings):
        self.reference_paths = reference_paths
        self.target_paths = target_paths
        self.illumination_settings = illumination_settings
        self.color_settings = color_settings
        
    def get_images(self):
        return {
            "reference_tile": self.ref_img,
            "target_tile": self.tar_img,
            "reference_white": self.ref_white,
            "target_white": self.tar_white,
            "reference_calibration": self.ref_cal,
            "target_calibration": self.target_cal
        }

    def load_images(self):
        self.ref_img = cv2.imread(self.reference_paths['tile'])
        self.tar_img = cv2.imread(self.target_paths['tile'])
        
        self.ref_white = cv2.imread(self.reference_paths['white'])
        self.tar_white = cv2.imread(self.target_paths['white'])
        
        self.ref_cal = cv2.imread(self.reference_paths['cal'])
        self.target_cal = cv2.imread(self.target_paths['cal'])
        
        if any(img is None for img in [self.ref_img, self.tar_img, self.ref_white, self.tar_white, self.ref_cal, self.target_cal]):
            raise ValueError("One or more images could not be loaded. Please check file paths.")
    
    def detect_sample_region(self, image, reference_area=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        v = np.median(blurred)
        lower = int(max(0, 0.66 * v))
        upper = int(min(255, 1.33 * v))
        edges = cv2.Canny(blurred, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.title("Blurred Gray")
        # plt.imshow(blurred, cmap='gray')
        # plt.axis('off')
        
        # plt.subplot(1, 3, 2)
        # plt.title(f"Canny Edges (thresh {lower}-{upper})")
        # plt.imshow(edges, cmap='gray')
        # plt.axis('off')
        
        # image_contours = image.copy()
        
        # cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
        # plt.subplot(1, 3, 3)
        # plt.title(f"Contours ({len(contours)})")
        # plt.imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if reference_area is not None and (area < reference_area * 0.8 or area > reference_area * 1.2):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) in [4]:
                return approx.reshape(-1, 2), area
        
        return None
    
    def detect_sample_region_cal(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            return None
        
        horizontals = []
        verticals = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 10:
                horizontals.append(line)
            elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
                verticals.append(line)
        
        horizontals = sorted(horizontals, key=lambda l: min(l[1], l[3]))
        top_line = horizontals[0]
        bottom_line = horizontals[-1]

        verticals = sorted(verticals, key=lambda l: min(l[0], l[2]))
        left_line = verticals[0]
        right_line = verticals[-1]

        def line_to_params(line):
            x1, y1, x2, y2 = line
            a = y2 - y1
            b = x1 - x2
            c = x2*y1 - x1*y2
            return a, b, c
        
        def intersection(l1, l2):
            a1, b1, c1 = line_to_params(l1)
            a2, b2, c2 = line_to_params(l2)
            det = a1*b2 - a2*b1
            if abs(det) < 1e-10:
                return None
            x = (b1*c2 - b2*c1) / det
            y = (c1*a2 - c2*a1) / det
            return int(x), int(y)

        tl = intersection(top_line, left_line)
        tr = intersection(top_line, right_line)
        bl = intersection(bottom_line, left_line)
        br = intersection(bottom_line, right_line)

        if None in [tl, tr, bl, br]:
            return None
        
        corners = np.array([tl, tr, br, bl], dtype=np.float32)

        return corners

    def preview_sample_region(self, corners, image, cropped, resize_to=(800, 600), show=True):
        image_resized = cv2.resize(image.copy(), resize_to)

        scale_x = resize_to[0] / image.shape[1]
        scale_y = resize_to[1] / image.shape[0]
        scaled_corners = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in corners])

        for pt in scaled_corners:
            cv2.circle(image_resized, tuple(pt), 6, (0, 0, 255), -1)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
        plt.title("Original with Detected Corners")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Region")
        plt.axis("off")
        plt.tight_layout()
        
        if show:
            plt.show()
        
    def visualize_3images(self, imgs, titles, resize_to=(800, 600), show=True):
        img0_resized = cv2.resize(imgs[0], resize_to)
        img1_resized = cv2.resize(imgs[1], resize_to)
        img2_resized = cv2.resize(imgs[2], resize_to)

        img0_rgb = cv2.cvtColor(img0_resized, cv2.COLOR_BGR2RGB)
        img1_rgb = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img0_rgb)
        plt.title(titles[0])
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(img1_rgb)
        plt.title(titles[1])
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(img2_rgb)
        plt.title(titles[2])
        plt.axis('off')

        plt.tight_layout()
        if show:
            plt.show()

    def run(self):
        self.load_images()
        print("Images loaded successfully.")
        
        self.illumination_corrector = IlluminationCorrector(self)
        self.tar_cal_illumination, self.tar_img_illumination = self.illumination_corrector.process()

        self.color_corrector = ColorCorrector(self)
        self.color_corrector.process()
        
        
if __name__ == "__main__":
    
    REF_CONDTION = '0_ls7'
    TARGET_CONDTION = '4_ls2'
    WHITE_DIR = './white'
    CAL_DIR = './cal'
    TILES_DIR = './tile images'
    
    reference_paths = {
        "cal": f'{CAL_DIR}/{REF_CONDTION}.jpg', 
        "tile": f'{TILES_DIR}/{REF_CONDTION}.jpg',
        "white": f'{WHITE_DIR}/{REF_CONDTION}.jpg'
    }
    target_paths = {
        "cal": f'{CAL_DIR}/{TARGET_CONDTION}.jpg',
        "tile": f'{TILES_DIR}/{TARGET_CONDTION}.jpg',
        "white": f'{WHITE_DIR}/{TARGET_CONDTION}.jpg'
    }
    illumination_settings = {
        "crop_percentage": 0.2,
        "smoothing_sigma": 301,
        "sample_step": 50,
        "region_size": 10,
        "interpolation_method": "cubic",  # 'rbf', 'linear', 'cubic', 'nearest'
        "visualize": False
    }
    
    color_settings = {
        "grid_shape": (4, 7),
        "sample_size": 50,
        "visualize": True
    }

    pipeline = ImageCorrectionPipeline(reference_paths, target_paths, illumination_settings, color_settings)
    pipeline.run()
