import streamlit as st
import numpy as np
import os
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pydicom as pyd
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import shift as ndimage_shift
import io
from datetime import datetime

# Page configuration
st.set_page_config(page_title="DICOM Analysis Tool", layout="wide")

# Initialize session state for storing results
if 'volume_left' not in st.session_state:
    st.session_state.volume_left = None
if 'volume_right' not in st.session_state:
    st.session_state.volume_right = None
if 'uptake_4h' not in st.session_state:
    st.session_state.uptake_4h = None
if 'uptake_24h' not in st.session_state:
    st.session_state.uptake_24h = None
if 'pixel_spacing' not in st.session_state:
    st.session_state.pixel_spacing = None
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = None
if 'patient_birth_date' not in st.session_state:
    st.session_state.patient_birth_date = None
if 'acquisition_dates' not in st.session_state:
    st.session_state.acquisition_dates = None
if 'manual_volume_left' not in st.session_state:
    st.session_state.manual_volume_left = None
if 'manual_volume_right' not in st.session_state:
    st.session_state.manual_volume_right = None

# Title
st.title("DICOM Medical Physics Analysis Tool")
st.markdown("Comprehensive analysis for volume images and thyroid uptake calculations")

# Sidebar for parameters
st.sidebar.header("Analysis Parameters")

# Analysis mode selection
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Volume & Ellipse Fitting", "Thyroid Uptake Analysis", "Dosimetry Calculations"]
)

st.sidebar.markdown("---")

# Common parameters
if analysis_mode == "Volume & Ellipse Fitting":
    st.sidebar.subheader("Volume Analysis Parameters")
    # threshold_volume = st.sidebar.slider(
    #     "Volume threshold (fraction of max)", 
    #     min_value=0.05, 
    #     max_value=0.99, 
    #     value=0.25, 
    #     step=0.01
    # )
    # profile_smoothing = st.sidebar.slider(
    #     "Profile smoothing (σ)", 
    #     min_value=1, 
    #     max_value=15, 
    #     value=5, 
    #     step=1
    # )
    # smooth_kernel_size = st.sidebar.slider(
    #     "Smoothing kernel size", 
    #     min_value=3, 
    #     max_value=7, 
    #     value=3, 
    #     step=2
    # )
    
    # local_mean_radius = st.sidebar.slider(
    #     "Local mean square side length (pixels)", 
    #     min_value=1, 
    #     max_value=10, 
    #     value=4, 
    #     step=1,
    # )


    threshold_volume = st.sidebar.slider(
    "Volume threshold (fraction of max)", 
    min_value=0.05, max_value=0.99, value=0.25, step=0.01,
    help="Sets the threshold for pixel inclusion based on maximum intensity. **Default: 0.25**"
    )

    profile_smoothing = st.sidebar.slider(
        "Profile smoothing (σ)", 
        min_value=1, max_value=15, value=5, step=1,
        help="Standard deviation for Gaussian kernel used to smooth the horizontal profile for lobe splitting. The only purpose is to correctly split the lobes. **Default: 5**"
    )

    smooth_kernel_size = st.sidebar.slider(
        "Smoothing kernel size", 
        min_value=3, max_value=7, value=3, step=2,
        help="Size of the 2D kernel for image noise reduction. **Default: 3**"
    )

    padding_width = st.sidebar.slider(
        "Padding width (pixels)", 
        min_value=0, 
        max_value=100, 
        value=50, 
        step=10,
        help="Size of pixel padding added to make the lobe appear more central in the image after separation. **Default: 50**"

    )

    local_mean_radius = st.sidebar.slider(
        "Local mean square side length (pixels)", 
        min_value=1, max_value=10, value=4, step=1,
        help="The side length of the square ROI used to calculate peak intensity for thresholding. **Default: 4**"
    )
    
elif analysis_mode == "Thyroid Uptake Analysis":
    st.sidebar.subheader("Thyroid Uptake Parameters")
    threshold_percentage = st.sidebar.slider(
        "Threshold (% of max intensity)", 
        min_value=0.01, 
        max_value=0.99, 
        value=0.15, 
        step=0.01
    )
    expansion_x = st.sidebar.slider(
        "Rectangle expansion X", 
        min_value=0.0, 
        max_value=3.0, 
        value=0.8, 
        step=0.05
    )
    expansion_y = st.sidebar.slider(
        "Rectangle expansion Y", 
        min_value=0.0, 
        max_value=3.0, 
        value=0.8, 
        step=0.05
    )
    mask_opacity = st.sidebar.slider(
        "Mask opacity", 
        min_value=0.01, 
        max_value=1.0, 
        value=0.5, 
        step=0.5
    )

    halflife_hours_str = st.sidebar.text_input(
        "Isotope half-life (hours)", 
        value="192.5976"
    )

    try:
        halflife_hours = float(halflife_hours_str)
    except ValueError:
        st.sidebar.error("Please enter a valid number for half-life.")
        halflife_hours = 192.5976

elif analysis_mode == "Dosimetry Calculations":
    st.sidebar.subheader("Dosimetry Parameters")
    target_dose_gy = st.sidebar.number_input(
        "Target dose (Gy)", 
        min_value=1.0, 
        max_value=10000.0, 
        value=120.0, 
        step=1.0
    )
    conversion_factor = st.sidebar.number_input(
        "Conversion factor", 
        min_value=0.1, 
        max_value=20.0, 
        value=5.829, 
        step=0.001,
        format="%.3f"
    )
    activity_divisor = st.sidebar.number_input(
        "Activity divisor", 
        min_value=1.0, 
        max_value=500.0, 
        value=132.0, 
        step=1.0
    )

# ==================== VOLUME ANALYSIS FUNCTIONS ====================

def create_gaussian_kernel(size):
    """Create normalized Gaussian kernel"""
    if size == 3:
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    elif size == 5:
        kernel = np.array([[1, 4, 6, 4, 1],
                          [4, 16, 24, 16, 4],
                          [6, 24, 36, 24, 6],
                          [4, 16, 24, 16, 4],
                          [1, 4, 6, 4, 1]])
    else:  # size == 7
        kernel = np.array([[1, 6, 15, 20, 15, 6, 1],
                          [6, 36, 90, 120, 90, 36, 6],
                          [15, 90, 225, 300, 225, 90, 15],
                          [20, 120, 300, 400, 300, 120, 20],
                          [15, 90, 225, 300, 225, 90, 15],
                          [6, 36, 90, 120, 90, 36, 6],
                          [1, 6, 15, 20, 15, 6, 1]])
    return kernel / kernel.sum()

def find_min_between_maxima_robust(profile):
    """Find minimum between two highest maxima"""
    profile = np.array(profile)
    peaks, _ = find_peaks(profile)
    
    if len(peaks) < 2:
        return None, None, None
    
    peak_values = profile[peaks]
    sorted_indices = np.argsort(peak_values)[::-1]
    
    max1_idx = peaks[sorted_indices[0]]
    max2_idx = None
    
    for i in range(1, len(sorted_indices)):
        candidate = peaks[sorted_indices[i]]
        if abs(max1_idx - candidate) > 1:
            max2_idx = candidate
            break
    
    if max2_idx is None:
        return None, None, None
    
    if max1_idx > max2_idx:
        max1_idx, max2_idx = max2_idx, max1_idx
    
    search_range = slice(max1_idx + 1, max2_idx)
    if search_range.start >= search_range.stop:
        return None, None, None
    
    min_idx = np.argmin(profile[search_range]) + search_range.start
    return min_idx, max1_idx, max2_idx

def compute_local_mean(volume, center, radius):
    """
    Compute local mean around a center point within a square ROI of exact size `radius x radius`.
    
    Parameters
    ----------
    volume : np.ndarray
        2D image array
    center : tuple
        (row, col) coordinates of the center point
    radius : int
        Size of the square ROI (total size will be radius x radius)
    
    Returns
    -------
    float
        Mean value within the square ROI
    """
    r0, c0 = int(center[0]), int(center[1])
    
    rmin = max(0, r0 - radius//2)
    rmax = min(volume.shape[0], rmin + radius)
    
    cmin = max(0, c0 - radius//2)
    cmax = min(volume.shape[1], cmin + radius)
    
    patch = volume[rmin:rmax, cmin:cmax]
    return patch.mean() if patch.size > 0 else 0.0

def fit_ellipse_fast(mask):
    """Fit ellipse to binary mask using OpenCV"""
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    
    if len(cnt) < 5:
        raise ValueError("Not enough points to fit an ellipse")
    
    (x, y), (major, minor), angle = cv2.fitEllipse(cnt)
    return (x, y), major / 2, minor / 2, angle

def analyze_volume_image(dcm_data, threshold, smoothing_sigma, kernel_size, padding, radius):
    """Complete volume analysis pipeline"""
    volume_img = dcm_data.pixel_array
    
    kernel = create_gaussian_kernel(kernel_size)
    volume_img_smooth = ndi.convolve(volume_img, kernel)
    
    profile = volume_img_smooth.sum(0)
    original_profile = profile.copy()
    profile = ndi.gaussian_filter(profile, smoothing_sigma)
    
    idx_min, max1, max2 = find_min_between_maxima_robust(profile)
    
    if idx_min is None:
        return None, "Could not find suitable maxima pair in profile"
    
    half_sx = volume_img_smooth[:, :idx_min]
    half_dx = volume_img_smooth[:, idx_min:]
    
    peak_sx = np.unravel_index(np.argmax(half_sx), half_sx.shape)
    peak_dx = np.unravel_index(np.argmax(half_dx), half_dx.shape)
    peak_dx_full = (peak_dx[0], peak_dx[1] + idx_min)
    
    max_val_sx = compute_local_mean(volume_img_smooth, peak_sx, radius)
    max_val_dx = compute_local_mean(volume_img_smooth, peak_dx_full, radius)
    
    mask_sx = volume_img_smooth > threshold * max_val_sx
    mask_dx = volume_img_smooth > threshold * max_val_dx
    
    vol_sx = (volume_img_smooth * mask_sx)[:, :idx_min]
    vol_dx = (volume_img_smooth * mask_dx)[:, idx_min:]
    
    pad_width = ((0, 0), (padding, padding))
    vol_sx_padded = np.pad(vol_sx, pad_width, mode='constant', constant_values=0)
    vol_dx_padded = np.pad(vol_dx, pad_width, mode='constant', constant_values=0)
    
    try:
        center_sx, a_sx, b_sx, ang_sx = fit_ellipse_fast(vol_sx_padded > 0)
        center_dx, a_dx, b_dx, ang_dx = fit_ellipse_fast(vol_dx_padded > 0)
    except Exception as e:
        return None, f"Error fitting ellipse: {str(e)}"
    
    results = {
        'volume_img': volume_img,
        'volume_img_smooth': volume_img_smooth,
        'profile': profile,
        'original_profile': original_profile,
        'idx_min': idx_min,
        'max1': max1,
        'max2': max2,
        'mask_sx': mask_sx,
        'mask_dx': mask_dx,
        'vol_sx_padded': vol_sx_padded,
        'vol_dx_padded': vol_dx_padded,
        'ellipse_sx': (center_sx, a_sx, b_sx, ang_sx),
        'ellipse_dx': (center_dx, a_dx, b_dx, ang_dx),
        'max_val_sx': max_val_sx,
        'max_val_dx': max_val_dx,
        'peak_sx': peak_sx,
        'peak_dx_full': peak_dx_full
    }
    
    return results, None

def extract_pixel_spacing(dcm_data):
    """Extract pixel spacing from DICOM metadata"""
    try:
        if hasattr(dcm_data, 'PixelSpacing'):
            pixel_spacing = dcm_data.PixelSpacing
            return float(pixel_spacing[0]), float(pixel_spacing[1])
        else:
            return None, None
    except:
        return None, None

# ==================== THYROID UPTAKE FUNCTIONS ====================

def create_threshold_mask(image_data, threshold_percentage):
    """Create mask for points above threshold percentage of maximum"""
    threshold_value = np.max(image_data) * threshold_percentage
    mask = image_data > threshold_value
    return mask, threshold_value

def find_inscribed_rectangle(mask):
    """Find the largest inscribed rectangle in the mask"""
    y_coords, x_coords = np.where(mask)
    
    if len(x_coords) == 0 or len(y_coords) == 0:
        return np.zeros_like(mask, dtype=bool), (0, 0, 0, 0)
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    rectangle_mask = np.zeros_like(mask, dtype=bool)
    rectangle_mask[y_min:y_max + 1, x_min:x_max + 1] = True
    
    return rectangle_mask, (x_min, x_max, y_min, y_max)

def expand_rectangle(rectangle_coords, image_shape, expansion_x, expansion_y):
    """Expand rectangle by given percentages"""
    x_min, x_max, y_min, y_max = rectangle_coords
    
    width = x_max - x_min
    height = y_max - y_min
    
    expand_x_pixels = int(width * expansion_x)
    expand_y_pixels = int(height * expansion_y)
    
    x_min_new = max(0, x_min - expand_x_pixels // 2)
    x_max_new = min(image_shape[1] - 1, x_max + expand_x_pixels // 2)
    y_min_new = max(0, y_min - expand_y_pixels // 2)
    y_max_new = min(image_shape[0] - 1, y_max + expand_y_pixels // 2)
    
    expanded_mask = np.zeros(image_shape, dtype=bool)
    expanded_mask[y_min_new:y_max_new + 1, x_min_new:x_max_new + 1] = True
    
    return expanded_mask, (x_min_new, x_max_new, y_min_new, y_max_new)

def shift_mask_to_image_barycenter(image_data, mask):
    """Shift mask to align with image barycenter"""
    y_indices, x_indices = np.indices(image_data.shape)
    total_intensity = np.sum(image_data)
    
    if total_intensity == 0:
        y_c, x_c = np.array(image_data.shape) / 2
    else:
        y_c = np.sum(y_indices * image_data) / total_intensity
        x_c = np.sum(x_indices * image_data) / total_intensity
    
    y_mask_indices, x_mask_indices = np.where(mask)
    if len(x_mask_indices) == 0:
        return mask.copy(), (0, 0)
    
    y_mask_center = np.mean(y_mask_indices)
    x_mask_center = np.mean(x_mask_indices)
    
    shift_y = y_c - y_mask_center
    shift_x = x_c - x_mask_center
    
    mask_shifted = ndimage_shift(mask.astype(float), shift=(shift_y, shift_x), 
                                 order=0, mode='constant', cval=0)
    mask_shifted = mask_shifted > 0.5
    
    return mask_shifted, (shift_y, shift_x)

def compute_rawintden_centered(image_data, mask3):
    """Compute RawIntDen with centered mask"""
    mask3_shifted, shift_values = shift_mask_to_image_barycenter(image_data, mask3)
    roi_pixels = image_data[mask3_shifted]
    raw_intden = np.sum(roi_pixels)
    
    return raw_intden, mask3_shifted, shift_values, roi_pixels

def calculate_decay_factor(hours_passed, halflife):
    """Calculate radioactive decay factor"""
    lambd = np.log(2) / halflife
    decadimento = np.exp(-lambd * hours_passed)
    return decadimento

def calculate_ellipsoid_volume(a, b):
    """
    Calculate volume of an ellipsoid with semi-axes a, b, b
    Volume = 4/3 * π * a * b * b
    
    Parameters
    ----------
    a : float
        Semi-major axis
    b : float
        Semi-minor axis
    
    Returns
    -------
    float
        Volume of the ellipsoid
    """
    return (4.0 / 3.0) * np.pi * a * a * b

def convert_pixels_to_mm(value_pixels, pixel_spacing_mm):
    """Convert pixel measurement to mm"""
    return value_pixels * pixel_spacing_mm

def convert_mm3_to_ml(volume_mm3):
    """Convert mm³ to mL (1 mL = 1000 mm³)"""
    return volume_mm3 / 1000.0

def extract_patient_info(dcm_data):
    """Extract patient information from DICOM metadata"""
    try:
        patient_name = str(dcm_data.PatientName) if hasattr(dcm_data, 'PatientName') else "Unknown"
        # Clean patient name: replace ^ with space (DICOM format: SURNAME^NAME -> SURNAME NAME)
        patient_name = patient_name.replace('^', ' ')

        # Extract birth date
        if hasattr(dcm_data, 'PatientBirthDate'):
            birth_date = dcm_data.PatientBirthDate
            # Format YYYYMMDD to DD/MM/YYYY
            if len(birth_date) == 8:
                birth_date = f"{birth_date[6:8]}/{birth_date[4:6]}/{birth_date[0:4]}"
        else:
            birth_date = "Unknown"

        # Extract acquisition date
        acquisition_date = None
        if hasattr(dcm_data, 'AcquisitionDate'):
            acq_date = dcm_data.AcquisitionDate
            # Format YYYYMMDD to DD/MM/YYYY
            if len(acq_date) == 8:
                acquisition_date = f"{acq_date[6:8]}/{acq_date[4:6]}/{acq_date[0:4]}"
        elif hasattr(dcm_data, 'StudyDate'):
            study_date = dcm_data.StudyDate
            if len(study_date) == 8:
                acquisition_date = f"{study_date[6:8]}/{study_date[4:6]}/{study_date[0:4]}"

        return patient_name, birth_date, acquisition_date
    except Exception as e:
        return "Unknown", "Unknown", None

def generate_pdf_report(fisico, nome, data_nascita, data_acquisizione,
                       captazione_4h, captazione_24h, volume,
                       attivita_120, attivita_200):
    """
    Generate PDF report based on the template from report.py

    Parameters:
    -----------
    fisico : str
        Name of the physicist
    nome : str
        Patient name
    data_nascita : str
        Patient birth date (format: DD/MM/YYYY)
    data_acquisizione : str
        Acquisition dates (format: DD-DD/MM/YYYY or similar)
    captazione_4h : float
        4-hour uptake percentage
    captazione_24h : float
        24-hour uptake percentage
    volume : float
        Total volume in mL
    attivita_120 : float
        Activity for 120 Gy in MBq
    attivita_200 : float
        Activity for 200 Gy in MBq

    Returns:
    --------
    io.BytesIO
        PDF file as bytes buffer
    """
    margin = 2
    a4_width = 21
    a4_height = 29.7

    fig, ax = plt.subplots(figsize=(a4_width/2.54, a4_height/2.54), facecolor='white', dpi=300)
    plt.xlim(0, a4_width)
    plt.ylim(0, a4_height)

    # Add title using annotate
    ax.annotate(
        'S.C. di FISICA SANITARIA – ASUGI\n'+
        'RISULTATI DELLA VALUTAZIONE DELLA CURVA DI CAPTAZIONE\n'+
        'E DEL VOLUME TIROIDEO',
        xy=(a4_width / 2, a4_height - margin),
        ha='center', va='top',
        fontsize=10.5, fontweight='bold'
    )

    # ax.annotate(
    #     f'Paziente: {nome} ({data_nascita})\n\n'+
    #     f'Date acquisizione immagini: {data_acquisizione}\n\n'+
    #     '\n\n\n'+
    #     'L'analisi delle immagini ha fornito i seguenti risultati: \n\n'+'\n'+
    #     f'        Captazione 4° ora = {int(captazione_4h):01d} %\n\n'+
    #     f'        Captazione 24° ora = {int(captazione_24h):01d} %\n\n'+
    #     f'        Volume di captazione: {int(volume):01d} ml\n\n' +
    #     '\n\n\n'+
    #     f'Attività da somministrare per 120 Gy = {int(attivita_120):01d} MBq\n\n'+
    #     f'Attività da somministrare per 200 Gy = {int(attivita_200):01d} MBq\n\n',
    #     xy=(margin+1, a4_height-6),
    #     ha='left', va='top',
    #     fontsize=8.5, color='k',
    # )


    ax.annotate(
        f'Paziente: {nome} ({data_nascita})\n\n'+
        f'Date acquisizione immagini: {data_acquisizione}\n\n'+
        '\n\n\n'+
        "L'analisi delle immagini ha fornito i seguenti risultati:\n\n\n"+
        f'        Captazione 4° ora = {round(captazione_4h):01d} %\n\n'+
        f'        Captazione 24° ora = {round(captazione_24h):01d} %\n\n'+
        f'        Volume di captazione: {round(volume):01d} ml\n\n' +
        '\n\n\n'+
        f'Attività da somministrare per 120 Gy = {round(attivita_120):01d} MBq\n\n'+
        f'Attività da somministrare per 200 Gy = {round(attivita_200):01d} MBq\n\n',
        xy=(margin+1, a4_height-6),
        ha='left', va='top',
        fontsize=8.5, color='k',
    )




    # ax.annotate(
    # 'Le valutazioni ed i calcoli sono stati svolti in accordo con le linee guida AIMN - SIE\n'+
    # ' - AIFM: "Il trattamento radiometabolico dell'ipertiroidismo.',
    # xy=(a4_width / 2, margin + 7),
    # ha='center', va='top',
    # fontsize=8.5, color='k',
    # style='italic'
    # )   


    ax.annotate(
    'Le valutazioni ed i calcoli sono stati svolti in accordo con le linee guida AIMN - SIE\n'+
    ' - AIFM: “Il trattamento radiometabolico dell’ipertiroidismo.',
    xy=(a4_width / 2, margin + 7),
    ha='center', va='top',
    fontsize=8.5, color='k',
    style='italic'
    )

    # Add physicist signature
    ax.annotate(
        'Il Fisico\n'+
        f'{fisico}\n',
        xy=(2 * a4_width / 3, margin + 3.5),
        ha='center', va='top',
        fontsize=9, color='k'
    )

    # Add border rectangle
    rect = patches.Rectangle((margin, margin+5), a4_width-2*margin, a4_height-2*(margin+4),
                             linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    # Save to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='pdf', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()

    return buf

# ==================== MAIN APP ====================

if analysis_mode == "Volume & Ellipse Fitting":
    st.header("Volume Image Analysis with Ellipse Fitting")
    
    uploaded_file = st.file_uploader("Upload DICOM volume file", type=['dcm'])
    
    if uploaded_file is not None:
        try:
            dcm = pyd.dcmread(uploaded_file)
            
            pixel_spacing_row, pixel_spacing_col = extract_pixel_spacing(dcm)
            if pixel_spacing_row is not None:
                st.info(f"Pixel spacing detected: {pixel_spacing_row:.3f} mm × {pixel_spacing_col:.3f} mm")
                st.session_state.pixel_spacing = (pixel_spacing_row, pixel_spacing_col)
            else:
                st.warning("Could not extract pixel spacing from DICOM. Calculations requiring real dimensions will not be available.")
                st.session_state.pixel_spacing = None
            
            with st.spinner("Analyzing volume image..."):
                results, error = analyze_volume_image(
                    dcm, 
                    threshold_volume, 
                    profile_smoothing, 
                    smooth_kernel_size, 
                    padding_width, 
                    local_mean_radius
                )
            
            if error:
                st.error(error)
            else:
                st.subheader("Original and Processed Images")
                
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    ax1.imshow(results['volume_img'], cmap='gray')
                    ax1.set_title('Original Volume Image')
                    ax1.axis('off')
                    st.pyplot(fig1)
                    plt.close()
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    im = ax2.imshow(results['volume_img_smooth'], cmap='turbo')
                    ax2.set_title('Smoothed Image')
                    ax2.axis('off')
                    plt.colorbar(im, ax=ax2, shrink=0.7)
                    st.pyplot(fig2)
                    plt.close()
                
                with col3:
                    fig3, ax3 = plt.subplots(figsize=(6, 5))
                    combined_mask = np.logical_or(results['mask_sx'], results['mask_dx'])
                    im = ax3.imshow(results['volume_img_smooth'] * combined_mask, cmap='turbo')
                    ax3.axvline(results['idx_min'], ls='--', color='white', lw=2)
                    ax3.set_title('Masked Regions')
                    ax3.axis('off')
                    plt.colorbar(im, ax=ax3, shrink=0.7)
                    st.pyplot(fig3)
                    plt.close()
                
                st.warning("Warning: the system of reference is the standard radiological (e.g. left of the image is right of the patient, called \"right\" lobe here)")
                st.subheader("Profile Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig4, ax4 = plt.subplots(figsize=(8, 4))
                    ax4.plot(results['original_profile'], label='Original', alpha=0.6)
                    ax4.plot(results['profile'], label='Smoothed', linewidth=2)
                    ax4.axvline(results['idx_min'], ls='--', color='red', 
                               label=f'Split at {results["idx_min"]}')
                    ax4.axvline(results['max1'], ls=':', color='green', alpha=0.7, label='Max 1')
                    ax4.axvline(results['max2'], ls=':', color='blue', alpha=0.7, label='Max 2')
                    ax4.set_title('Horizontal Profile')
                    ax4.set_xlabel('Column index')
                    ax4.set_ylabel('Summed intensity')
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    st.pyplot(fig4)
                    plt.close()
                
                with col2:
                    #st.metric("Split Index", results['idx_min'])
                    st.metric("Max Right Value", f"{results['max_val_sx']:.2f}")
                    st.metric("Max Left Value", f"{results['max_val_dx']:.2f}")
                    #st.metric("Peak Left (row, col)", f"{results['peak_sx']}")
                    #st.metric("Peak Right (row, col)", f"{results['peak_dx_full']}")
                
                st.subheader("Thresholded Regions")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig5, ax5 = plt.subplots(figsize=(6, 5))
                    im = ax5.imshow(results['volume_img_smooth'] * results['mask_sx'], cmap='turbo')
                    ax5.axvline(results['idx_min'], ls='--', color='white', lw=2)
                    ax5.set_title(f'Right Threshold >{threshold_volume * results["max_val_sx"]:.2f}')
                    plt.axvspan(results['idx_min'], results['volume_img_smooth'].shape[1] , alpha=0.70, color='white')
                    ax5.axis('off')
                    plt.colorbar(im, ax=ax5, shrink=0.7)
                    st.pyplot(fig5)
                    plt.close()
                
                with col2:
                    fig6, ax6 = plt.subplots(figsize=(6, 5))
                    im = ax6.imshow(results['volume_img_smooth'] * results['mask_dx'], cmap='turbo')
                    ax6.axvline(results['idx_min'], ls='--', color='white', lw=2)
                    ax6.set_title(f'Left Threshold >{threshold_volume * results["max_val_dx"]:.2f}')
                    plt.axvspan(0, results['idx_min'], alpha=0.70, color='white')
                    ax6.axis('off')
                    plt.colorbar(im, ax=ax6, shrink=0.7)
                    st.pyplot(fig6)
                    plt.close()
                
                st.subheader("Ellipse Fitting Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig7, ax7 = plt.subplots(figsize=(6, 5))
                    ax7.imshow(results['vol_sx_padded'], cmap='gray')
                    center_sx, a_sx, b_sx, ang_sx = results['ellipse_sx']
                    ellipse_sx = Ellipse(center_sx, 2*a_sx, 2*b_sx, angle=ang_sx, 
                                        fill=False, color='red', lw=2)
                    ax7.add_patch(ellipse_sx)
                    ax7.set_title('Right Ellipse Fit')
                    ax7.axis('off')
                    st.pyplot(fig7)
                    plt.close()
                    
                    st.write(f"**Center:** ({center_sx[0]:.1f}, {center_sx[1]:.1f})")
                    st.write(f"**Semi-major axis (a):** {a_sx:.1f} px")
                    st.write(f"**Semi-minor axis (b):** {b_sx:.1f} px")
                    st.write(f"**Angle:** {ang_sx:.1f}°")
                    
                    spacing = st.session_state.pixel_spacing[0]
                    a_sx_mm = convert_pixels_to_mm(a_sx, spacing)
                    b_sx_mm = convert_pixels_to_mm(b_sx, spacing)
                    vol_sx_ml = convert_mm3_to_ml(calculate_ellipsoid_volume(a_sx_mm, b_sx_mm))
                    st.write(f"**Estimated Volume:** {vol_sx_ml:.3f} ml") # BOOKMARK1

                    if st.button("💾 Save Right Volume Data", key="save_left"):
                        st.session_state.volume_left = {
                            'a': a_sx,
                            'b': b_sx,
                            'angle': ang_sx,
                            'center': center_sx
                        }
                        st.success("Left volume data saved!")
                
                with col2:
                    fig8, ax8 = plt.subplots(figsize=(6, 5))
                    ax8.imshow(results['vol_dx_padded'], cmap='gray')
                    center_dx, a_dx, b_dx, ang_dx = results['ellipse_dx']
                    ellipse_dx = Ellipse(center_dx, 2*a_dx, 2*b_dx, angle=ang_dx,
                                        fill=False, color='red', lw=2)
                    ax8.add_patch(ellipse_dx)
                    ax8.set_title('Left Ellipse Fit')
                    ax8.axis('off')
                    st.pyplot(fig8)
                    plt.close()

                    st.write(f"**Center:** ({center_dx[0]:.1f}, {center_dx[1]:.1f})")
                    st.write(f"**Semi-major axis (a):** {a_dx:.1f} px")
                    st.write(f"**Semi-minor axis (b):** {b_dx:.1f} px")
                    st.write(f"**Angle:** {ang_dx:.1f}°") # BOOKMARK1

                    spacing = st.session_state.pixel_spacing[0]
                    a_dx_mm = convert_pixels_to_mm(a_dx, spacing)
                    b_dx_mm = convert_pixels_to_mm(b_dx, spacing)
                    vol_dx_ml = convert_mm3_to_ml(calculate_ellipsoid_volume(a_dx_mm, b_dx_mm))
                    st.write(f"**Estimated Volume:** {vol_dx_ml:.3f} ml") # BOOKMARK1


                    if st.button("💾 Save Left Volume Data", key="save_right"):
                        st.session_state.volume_right = {
                            'a': a_dx,
                            'b': b_dx,
                            'angle': ang_dx,
                            'center': center_dx
                        }
                        st.success("Right volume data saved!")

                # Manual Volume Override Section
                st.markdown("---")
                st.subheader("Manual Volume Override (Optional)")
                st.info("For difficult cases where automatic ellipse fitting is unreliable, you can manually enter the volume (in mL) for either lobe. Mix and match: use ellipse fit for one side and manual entry for the other!")

                col_manual_left, col_manual_right = st.columns(2)

                with col_manual_left:
                    st.markdown("**Right Manual Volume**")
                    use_manual_left = st.checkbox(
                        "Override Right with manual volume",
                        key="use_manual_left_check",
                        help="Use manual volume instead of ellipse fit for right lobe"
                    )

                    if use_manual_left:
                        manual_vol_left = st.number_input(
                            "Right volume (mL)",
                            min_value=0.1,
                            max_value=250.0,
                            value=st.session_state.manual_volume_left if st.session_state.manual_volume_left else 5.0,
                            step=0.1,
                            format="%.2f",
                            key="manual_vol_left_input",
                            help="Enter the manually estimated right lobe volume in mL"
                        )

                        if st.button("💾 Save Right Manual Volume", key="save_manual_vol_left"):
                            st.session_state.manual_volume_left = manual_vol_left
                            st.session_state.volume_left = None  # Clear ellipse data for this side
                            st.success(f"Right manual volume saved: {manual_vol_left:.2f} mL")
                    else:
                        if st.session_state.manual_volume_left is not None:
                            if st.button("🗑️ Clear Right Manual Volume", key="clear_manual_left"):
                                st.session_state.manual_volume_left = None
                                st.info("Right manual volume cleared")

                with col_manual_right:
                    st.markdown("**Left Manual Volume**")
                    use_manual_right = st.checkbox(
                        "Override Left with manual volume",
                        key="use_manual_right_check",
                        help="Use manual volume instead of ellipse fit for left lobe"
                    )

                    if use_manual_right:
                        manual_vol_right = st.number_input(
                            "Left volume (mL)",
                            min_value=0.0,
                            max_value=999.0,
                            value=st.session_state.manual_volume_right if st.session_state.manual_volume_right else 5.0,
                            step=0.1,
                            format="%.2f",
                            key="manual_vol_right_input",
                            help="Enter the manually estimated left lobe volume in mL"
                        )

                        if st.button("💾 Save Left Manual Volume", key="save_manual_vol_right"):
                            st.session_state.manual_volume_right = manual_vol_right
                            st.session_state.volume_right = None  # Clear ellipse data for this side
                            st.success(f"Left manual volume saved: {manual_vol_right:.2f} mL")
                    else:
                        if st.session_state.manual_volume_right is not None:
                            if st.button("🗑️ Clear Left Manual Volume", key="clear_manual_right"):
                                st.session_state.manual_volume_right = None
                                st.info("Left manual volume cleared")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif analysis_mode == "Thyroid Uptake Analysis":
    st.header("Thyroid Uptake Analysis")
    
    st.markdown("""
    Upload DICOM files for thyroid uptake calculation. The app expects:
    - Reference file (Tiroide) to create the ROI mask
    - Thyroid (Tiroide) images at 4h and 24h
    - Background (Fondo) images at 4h and 24h
    - Phantom (Fantoccio) images at 4h and 24h
    """)
    
    st.subheader("Upload Thyroid Reference File")
    reference_file = st.file_uploader("Reference DICOM (for mask creation)", type=['dcm'], key='ref')
    
    st.subheader("Upload 4-hour Images")
    col1, col2, col3 = st.columns(3)
    with col1:
        thyroid_4h = st.file_uploader("Thyroid 4H", type=['dcm'], key='thy4')
    with col2:
        background_4h = st.file_uploader("Background 4H", type=['dcm'], key='bg4')
    with col3:
        phantom_4h = st.file_uploader("Phantom 4H", type=['dcm'], key='ph4')
    
    st.subheader("Upload 24-hour Images")
    col1, col2, col3 = st.columns(3)
    with col1:
        thyroid_24h = st.file_uploader("Thyroid 24H", type=['dcm'], key='thy24')
    with col2:
        background_24h = st.file_uploader("Background 24H", type=['dcm'], key='bg24')
    with col3:
        phantom_24h = st.file_uploader("Phantom 24H", type=['dcm'], key='ph24')
    
    if reference_file is not None:
        try:
            ref_dcm = pyd.dcmread(reference_file)
            ref_image = ref_dcm.pixel_array

            # Extract patient information from reference DICOM
            patient_name, birth_date, acq_date_ref = extract_patient_info(ref_dcm)
            st.session_state.patient_name = patient_name
            st.session_state.patient_birth_date = birth_date

            # Display patient info
            st.info(f"Patient: {patient_name} | Birth Date: {birth_date}")

            st.subheader("Reference Image and Mask Creation")
            
            mask1, threshold_value = create_threshold_mask(ref_image, threshold_percentage)
            mask2, rectangle_coords = find_inscribed_rectangle(mask1)
            mask3, expanded_coords = expand_rectangle(
                rectangle_coords, ref_image.shape, expansion_x, expansion_y
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            def create_rgba_mask(mask, alpha=0.5):
                rgba = np.zeros((*mask.shape, 4))
                rgba[mask] = [1, 0, 0, alpha]
                return rgba
            
            with col1:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(ref_image, cmap='gray')
                ax.set_title('Reference Image')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(ref_image, cmap='gray')
                ax.imshow(create_rgba_mask(mask1, mask_opacity))
                ax.set_title(f'Threshold >{threshold_percentage*100:.0f}%')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            
            with col3:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(ref_image, cmap='gray')
                ax.imshow(create_rgba_mask(mask2, mask_opacity))
                ax.set_title('Inscribed Rectangle')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            
            with col4:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(ref_image, cmap='gray')
                ax.imshow(create_rgba_mask(mask3, mask_opacity))
                ax.set_title(f'Expanded +{expansion_x*100:.0f}%')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            
            st.info(f"Mask contains {np.sum(mask3)} pixels. Rectangle coords: x=[{expanded_coords[0]}, {expanded_coords[1]}], y=[{expanded_coords[2]}, {expanded_coords[3]}]")
            
            all_files_present = all([
                thyroid_4h, background_4h, phantom_4h,
                thyroid_24h, background_24h, phantom_24h
            ])
            
            if all_files_present:
                st.subheader("RawIntDen Calculations")

                # Extract acquisition dates from thyroid images
                thy4_dcm = pyd.dcmread(thyroid_4h)
                thy24_dcm = pyd.dcmread(thyroid_24h)

                _, _, acq_date_4h = extract_patient_info(thy4_dcm)
                _, _, acq_date_24h = extract_patient_info(thy24_dcm)

                # Format acquisition dates for report (e.g., "05-10/10/2022")
                if acq_date_4h and acq_date_24h:
                    # Extract day and month from both dates
                    # Format: DD/MM/YYYY
                    day_4h = acq_date_4h.split('/')[0]
                    day_24h = acq_date_24h.split('/')[0]
                    month = acq_date_4h.split('/')[1]
                    year = acq_date_4h.split('/')[2]
                    acquisition_dates_formatted = f"{day_4h}-{day_24h}/{month}/{year}"
                    st.session_state.acquisition_dates = acquisition_dates_formatted
                    st.info(f"Acquisition dates: {acquisition_dates_formatted}")

                thy4_img = thy4_dcm.pixel_array
                bg4_img = pyd.dcmread(background_4h).pixel_array
                ph4_img = pyd.dcmread(phantom_4h).pixel_array
                
                rawintden_thy4, mask_thy4, shift_thy4, roi_thy4 = compute_rawintden_centered(thy4_img, mask3)
                rawintden_bg4, mask_bg4, shift_bg4, roi_bg4 = compute_rawintden_centered(bg4_img, mask3)
                rawintden_ph4, mask_ph4, shift_ph4, roi_ph4 = compute_rawintden_centered(ph4_img, mask3)

                thy24_img = thy24_dcm.pixel_array
                bg24_img = pyd.dcmread(background_24h).pixel_array
                ph24_img = pyd.dcmread(phantom_24h).pixel_array
                
                rawintden_thy24, mask_thy24, shift_thy24, roi_thy24 = compute_rawintden_centered(thy24_img, mask3)
                rawintden_bg24, mask_bg24, shift_bg24, roi_bg24 = compute_rawintden_centered(bg24_img, mask3)
                rawintden_ph24, mask_ph24, shift_ph24, roi_ph24 = compute_rawintden_centered(ph24_img, mask3)
                
                st.markdown("### 4-Hour Acquisitions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(thy4_img, cmap='gray')
                    ax.imshow(create_rgba_mask(mask_thy4, mask_opacity))
                    ax.set_title('Thyroid 4H')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    st.metric("RawIntDen", f"{rawintden_thy4:.0f}")
                    st.caption(f"Shift: Δx={shift_thy4[1]:.1f}, Δy={shift_thy4[0]:.1f}")
                    st.caption(f"Mean intensity: {np.mean(roi_thy4):.2f}")
                
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(bg4_img, cmap='gray')
                    ax.imshow(create_rgba_mask(mask_bg4, mask_opacity))
                    ax.set_title('Background 4H')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    st.metric("RawIntDen", f"{rawintden_bg4:.0f}")
                    st.caption(f"Shift: Δx={shift_bg4[1]:.1f}, Δy={shift_bg4[0]:.1f}")
                    st.caption(f"Mean intensity: {np.mean(roi_bg4):.2f}")
                
                with col3:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(ph4_img, cmap='gray')
                    ax.imshow(create_rgba_mask(mask_ph4, mask_opacity))
                    ax.set_title('Phantom 4H')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    st.metric("RawIntDen", f"{rawintden_ph4:.0f}")
                    st.caption(f"Shift: Δx={shift_ph4[1]:.1f}, Δy={shift_ph4[0]:.1f}")
                    st.caption(f"Mean intensity: {np.mean(roi_ph4):.2f}")
                
                st.markdown("### 24-Hour Acquisitions")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(thy24_img, cmap='gray')
                    ax.imshow(create_rgba_mask(mask_thy24, mask_opacity))
                    ax.set_title('Thyroid 24H')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    st.metric("RawIntDen", f"{rawintden_thy24:.0f}")
                    st.caption(f"Shift: Δx={shift_thy24[1]:.1f}, Δy={shift_thy24[0]:.1f}")
                    st.caption(f"Mean intensity: {np.mean(roi_thy24):.2f}")
                
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(bg24_img, cmap='gray')
                    ax.imshow(create_rgba_mask(mask_bg24, mask_opacity))
                    ax.set_title('Background 24H')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    st.metric("RawIntDen", f"{rawintden_bg24:.0f}")
                    st.caption(f"Shift: Δx={shift_bg24[1]:.1f}, Δy={shift_bg24[0]:.1f}")
                    st.caption(f"Mean intensity: {np.mean(roi_bg24):.2f}")
                
                with col3:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(ph24_img, cmap='gray')
                    ax.imshow(create_rgba_mask(mask_ph24, mask_opacity))
                    ax.set_title('Phantom 24H')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
                    st.metric("RawIntDen", f"{rawintden_ph24:.0f}")
                    st.caption(f"Shift: Δx={shift_ph24[1]:.1f}, Δy={shift_ph24[0]:.1f}")
                    st.caption(f"Mean intensity: {np.mean(roi_ph24):.2f}")
                
                st.subheader("Radioactive Decay Corrections")
                
                decay_4h = calculate_decay_factor(4, halflife_hours)
                decay_24h = calculate_decay_factor(24, halflife_hours)
                
                phantom_4h_corrected = rawintden_ph4 / decay_4h
                phantom_24h_corrected = rawintden_ph24 / decay_24h
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Decay factor at 4h", f"{decay_4h:.6f}")
                    st.metric("Phantom 4H (decay corrected)", f"{phantom_4h_corrected:.0f}")
                
                with col2:
                    st.metric("Decay factor at 24h", f"{decay_24h:.6f}")
                    st.metric("Phantom 24H (decay corrected)", f"{phantom_24h_corrected:.0f}")
                
                variation = (phantom_24h_corrected - phantom_4h_corrected) / phantom_4h_corrected * 100
                st.info(f"Variation between decay-corrected phantoms: {variation:.2f}%")
                
                st.subheader("Thyroid Uptake Results")


                
                st.markdown("""
                The thyroid uptake is calculated as a percentage of the administered dose using the following methodology:
                """)
                
                # LaTeX formatted formula
                st.latex(r"Uptake\% = \left( \frac{RawIntDen_{Thyroid} - RawIntDen_{Background}}{RawIntDen_{PhantomCorrected}} \right) \times 100")
                
                st.markdown("""
                **Normalization Methods:**
                You can choose between two normalization values:
                * **4H Phantom Normalization:** Uses the phantom measured at 4 hours to normalize the uptake.
                * **24H Phantom Normalization:** Uses the phantom measured at 24 hours to normalize the uptake. 
                If the phantom acquisitions were performed without issues, the two value corrected for decay should ideally be the same.
                
                **Decay Correction:**
                The $PhantomCorrected$ value accounts for the radioactive decay of the isotope between the time of dose administration and the time of measurement (4h or 24h) using the provided half-life.
                """)
                
                st.markdown("#### Using Phantom 24H for normalization:")
                uptake_4h_with_24h = (rawintden_thy4 - rawintden_bg4) / phantom_24h_corrected * 100
                uptake_24h_with_24h = (rawintden_thy24 - rawintden_bg24) / phantom_24h_corrected * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Uptake at 4 hours", f"{uptake_4h_with_24h:.1f}%", 
                             delta=None, delta_color="off")
                with col2:
                    st.metric("Uptake at 24 hours", f"{uptake_24h_with_24h:.1f}%",
                             delta=None, delta_color="off")
                with col3:
                    if st.button("💾 Save Uptake (24H Phantom)", key="save_uptake_24h"):
                        st.session_state.uptake_4h = uptake_4h_with_24h
                        st.session_state.uptake_24h = uptake_24h_with_24h
                        st.success("Uptake data (24H phantom) saved!")
                
                st.markdown("#### Using Phantom 4H for normalization:")
                uptake_4h_with_4h = (rawintden_thy4 - rawintden_bg4) / phantom_4h_corrected * 100
                uptake_24h_with_4h = (rawintden_thy24 - rawintden_bg24) / phantom_4h_corrected * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Uptake at 4 hours", f"{uptake_4h_with_4h:.1f}%",
                             delta=None, delta_color="off")
                with col2:
                    st.metric("Uptake at 24 hours", f"{uptake_24h_with_4h:.1f}%",
                             delta=None, delta_color="off")
                with col3:
                    if st.button("💾 Save Uptake (4H Phantom)", key="save_uptake_4h"):
                        st.session_state.uptake_4h = uptake_4h_with_4h
                        st.session_state.uptake_24h = uptake_24h_with_4h
                        st.success("Uptake data (4H phantom) saved!")
                
                st.subheader("Summary Table")
                import pandas as pd
                
                summary_data = {
                    'Measurement': [
                        'Thyroid 4H', 'Background 4H', 'Phantom 4H',
                        'Thyroid 24H', 'Background 24H', 'Phantom 24H'
                    ],
                    'RawIntDen': [
                        rawintden_thy4, rawintden_bg4, rawintden_ph4,
                        rawintden_thy24, rawintden_bg24, rawintden_ph24
                    ],
                    'Mean Intensity': [
                        np.mean(roi_thy4), np.mean(roi_bg4), np.mean(roi_ph4),
                        np.mean(roi_thy24), np.mean(roi_bg24), np.mean(roi_ph24)
                    ],
                    'Num Pixels': [
                        np.sum(mask_thy4), np.sum(mask_bg4), np.sum(mask_ph4),
                        np.sum(mask_thy24), np.sum(mask_bg24), np.sum(mask_ph24)
                    ]
                }
                
                df = pd.DataFrame(summary_data)
                st.dataframe(df, use_container_width=True)
                
                st.subheader("Uptake Comparison Chart")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                time_points = ['4 hours', '24 hours']
                uptake_24h_norm = [uptake_4h_with_24h, uptake_24h_with_24h]
                uptake_4h_norm = [uptake_4h_with_4h, uptake_24h_with_4h]
                
                x = np.arange(len(time_points))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, uptake_24h_norm, width, 
                              label='Normalized with Phantom 24H', alpha=0.8)
                bars2 = ax.bar(x + width/2, uptake_4h_norm, width, 
                              label='Normalized with Phantom 4H', alpha=0.8)
                
                ax.set_ylabel('Uptake (%)')
                ax.set_title('Thyroid Uptake Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(time_points)
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%',
                               ha='center', va='bottom', fontsize=9)
                
                st.pyplot(fig)
                plt.close()
                
            else:
                st.warning("Please upload all required files (thyroid, background, and phantom at both 4h and 24h) to perform uptake calculations.")
        
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

elif analysis_mode == "Dosimetry Calculations":
    st.header("Dosimetry Calculations")
    
    st.markdown("""
    This section uses previously saved volume and uptake data to calculate the required activity for treatment.
    
    **Formula:** Activity (MBq) = (Conversion Factor × Target Dose × Total Volume) / (Max Uptake × Activity Divisor)
    
    Where:
    - Total Volume = Volume_left + Volume_right (computed from ellipsoid formula: 4/3 × π × a × b²)
    - Max Uptake = max(Uptake_4h, Uptake_24h) converted to decimal
    """)
    
    st.subheader("Saved Data Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Volume Data:**")

        # Check Right lobe
        if st.session_state.manual_volume_left is not None:
            st.success("✓ Right volume: Manual override")
            st.write(f"Volume: {st.session_state.manual_volume_left:.2f} mL")
        elif st.session_state.volume_left is not None:
            st.success("✓ Right volume: Ellipse fit")
            st.write(f"a = {st.session_state.volume_left['a']:.2f} px, b = {st.session_state.volume_left['b']:.2f} px")
        else:
            st.warning("✗ Right volume not saved")

        # Check Left lobe
        if st.session_state.manual_volume_right is not None:
            st.success("✓ Left volume: Manual override")
            st.write(f"Volume: {st.session_state.manual_volume_right:.2f} mL")
        elif st.session_state.volume_right is not None:
            st.success("✓ Left volume: Ellipse fit")
            st.write(f"a = {st.session_state.volume_right['a']:.2f} px, b = {st.session_state.volume_right['b']:.2f} px")
        else:
            st.warning("✗ Left volume not saved")

        # Check pixel spacing (only needed if at least one lobe uses ellipse fit)
        needs_pixel_spacing = (st.session_state.volume_left is not None or
                               st.session_state.volume_right is not None)
        if needs_pixel_spacing:
            if st.session_state.pixel_spacing is not None:
                st.success("✓ Pixel spacing available")
                st.write(f"Spacing: {st.session_state.pixel_spacing[0]:.3f} × {st.session_state.pixel_spacing[1]:.3f} mm")
            else:
                st.warning("✗ Pixel spacing not available (required for ellipse-based volumes)")
    
    with col2:
        st.markdown("**Uptake Data:**")
        if st.session_state.uptake_4h is not None:
            st.success("✓ Uptake data saved")
            st.write(f"4h: {st.session_state.uptake_4h:.2f}%")
            st.write(f"24h: {st.session_state.uptake_24h:.2f}%")
        else:
            st.warning("✗ Uptake data not saved")
    
    # Check which data are available for each lobe
    left_volume_available = (st.session_state.manual_volume_left is not None or
                             st.session_state.volume_left is not None)
    right_volume_available = (st.session_state.manual_volume_right is not None or
                              st.session_state.volume_right is not None)

    # Check if pixel spacing is available when needed (for ellipse-based calculations)
    needs_pixel_spacing = (st.session_state.volume_left is not None or
                           st.session_state.volume_right is not None)
    pixel_spacing_ok = (not needs_pixel_spacing or
                        st.session_state.pixel_spacing is not None)

    # Volume data is available if both lobes have data AND pixel spacing is ok
    volume_data_available = (left_volume_available and
                             right_volume_available and
                             pixel_spacing_ok)

    uptake_data_available = all([
        st.session_state.uptake_4h is not None,
        st.session_state.uptake_24h is not None
    ])

    all_data_available = volume_data_available and uptake_data_available

    # Import pandas if any data is available
    if volume_data_available or uptake_data_available:
        import pandas as pd

    # Initialize variables for later use
    total_volume_ml = None
    max_uptake_decimal = None
    max_uptake_percent = None

    # Display Volume Calculations if volume data is available
    if volume_data_available:
        st.subheader("Volume Calculations")

        # Calculate Right lobe volume
        if st.session_state.manual_volume_left is not None:
            # Manual override for right lobe
            volume_left_ml = st.session_state.manual_volume_left
            left_source = "Manual"
            left_details = "—"
        else:
            # Ellipse-based for right lobe
            a_left = st.session_state.volume_left['a']
            b_left = st.session_state.volume_left['b']
            pixel_spacing = st.session_state.pixel_spacing[0]
            a_left_mm = convert_pixels_to_mm(a_left, pixel_spacing)
            b_left_mm = convert_pixels_to_mm(b_left, pixel_spacing)
            volume_left_mm3 = calculate_ellipsoid_volume(a_left_mm, b_left_mm)
            volume_left_ml = convert_mm3_to_ml(volume_left_mm3)
            left_source = "Ellipse"
            left_details = f"a={a_left:.1f}px, b={b_left:.1f}px"

        # Calculate Left lobe volume
        if st.session_state.manual_volume_right is not None:
            # Manual override for left lobe
            volume_right_ml = st.session_state.manual_volume_right
            right_source = "Manual"
            right_details = "—"
        else:
            # Ellipse-based for left lobe
            a_right = st.session_state.volume_right['a']
            b_right = st.session_state.volume_right['b']
            if 'pixel_spacing' not in locals():
                pixel_spacing = st.session_state.pixel_spacing[0]
            a_right_mm = convert_pixels_to_mm(a_right, pixel_spacing)
            b_right_mm = convert_pixels_to_mm(b_right, pixel_spacing)
            volume_right_mm3 = calculate_ellipsoid_volume(a_right_mm, b_right_mm)
            volume_right_ml = convert_mm3_to_ml(volume_right_mm3)
            right_source = "Ellipse"
            right_details = f"a={a_right:.1f}px, b={b_right:.1f}px"

        # Calculate total volume
        total_volume_ml = volume_left_ml + volume_right_ml

        # Display summary table
        volume_data = {
            'Region': ['Right ', 'Left ', 'Total'],
            'Source': [left_source, right_source, '—'],
            'Details': [left_details, right_details, '—'],
            'Volume (mL)': [f"{volume_left_ml:.3f}", f"{volume_right_ml:.3f}", f"{total_volume_ml:.3f}"]
        }

        df_volume = pd.DataFrame(volume_data)
        st.dataframe(df_volume, use_container_width=True, hide_index=True)
        
        # # Display volume visualization
        # st.markdown("#### Volume Comparison")
        
        # fig_vol, ax_vol = plt.subplots(figsize=(6, 5))
        
        # regions = ['Left', 'Right']
        # volumes_ml = [volume_left_ml, volume_right_ml]
        # colors = ['#FF6B6B', '#4ECDC4']
        
        # ax_vol.bar(regions, volumes_ml, color=colors, alpha=0.7, edgecolor='black')
        # ax_vol.set_ylabel('Volume (mL)')
        # ax_vol.set_title('Volume Comparison by Region')
        # ax_vol.grid(True, alpha=0.3, axis='y')
        
        # for i, (region, vol) in enumerate(zip(regions, volumes_ml)):
        #     ax_vol.text(i, vol, f'{vol:.3f} mL', ha='center', va='bottom', fontweight='bold')
        
        # plt.tight_layout()
        # st.pyplot(fig_vol)
        # plt.close()
    
    # Display Uptake Summary if uptake data is available
    if uptake_data_available:
        st.subheader("Uptake Summary")
        
        uptake_4h_decimal = st.session_state.uptake_4h / 100.0
        uptake_24h_decimal = st.session_state.uptake_24h / 100.0
        max_uptake_decimal = max(uptake_4h_decimal, uptake_24h_decimal)
        max_uptake_percent = max(st.session_state.uptake_4h, st.session_state.uptake_24h)
        
        uptake_data = {
            'Time Point': ['4 hours', '24 hours', 'Maximum (used)'],
            'Uptake (%)': [
                f"{st.session_state.uptake_4h:.2f}",
                f"{st.session_state.uptake_24h:.2f}",
                f"{max_uptake_percent:.2f}"
            ],
            'Uptake (decimal)': [
                f"{uptake_4h_decimal:.4f}",
                f"{uptake_24h_decimal:.4f}",
                f"{max_uptake_decimal:.4f}"
            ]
        }
        
        df_uptake = pd.DataFrame(uptake_data)
        st.dataframe(df_uptake, use_container_width=True, hide_index=True)
        
        # # Display uptake visualization
        # st.markdown("#### Uptake Comparison")
        
        # fig_upt, ax_upt = plt.subplots(figsize=(6, 5))
        
        # time_points = ['4h', '24h']
        # uptakes = [st.session_state.uptake_4h, st.session_state.uptake_24h]
        # colors_uptake = ['#95E1D3', '#F38181']
        
        # bars = ax_upt.bar(time_points, uptakes, color=colors_uptake, alpha=0.7, edgecolor='black')
        # ax_upt.set_ylabel('Uptake (%)')
        # ax_upt.set_title('Thyroid Uptake by Time Point')
        # ax_upt.grid(True, alpha=0.3, axis='y')
        
        # for i, (time, uptake) in enumerate(zip(time_points, uptakes)):
        #     ax_upt.text(i, uptake, f'{uptake:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # max_idx = 0 if uptakes[0] > uptakes[1] else 1
        # bars[max_idx].set_edgecolor('red')
        # bars[max_idx].set_linewidth(3)
        
        # plt.tight_layout()
        # st.pyplot(fig_upt)
        # plt.close()
    
    # Display Activity Calculation only if all data is available
    if all_data_available:
        st.subheader("Activity Calculation")
        
        activity_mbq = (conversion_factor * target_dose_gy * total_volume_ml) / (max_uptake_decimal * activity_divisor)
        
        st.markdown(f"""
        **Formula:**
        ```
        Activity = (CF × Dose × Volume) / (Uptake × Divisor)
        Activity = ({conversion_factor:.3f} × {target_dose_gy:.1f} Gy × {total_volume_ml:.3f} mL) / ({max_uptake_decimal:.4f} × {activity_divisor:.1f})
        Activity = {activity_mbq:.2f} MBq
        ```
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Volume", f"{total_volume_ml:.3f} mL")
        
        with col2:
            st.metric("Max Uptake", f"{max_uptake_percent:.2f}%")
        
        with col3:
            st.metric("Required Activity", f"{activity_mbq:.2f} MBq", 
                     help="Activity needed to deliver the target dose")
        
        st.markdown("### Parameter Summary")

        # Build parameter data conditionally
        param_list = [
            ('Target Dose', f"{target_dose_gy:.1f} Gy"),
            ('Conversion Factor', f"{conversion_factor:.3f}"),
            ('Activity Divisor', f"{activity_divisor:.1f}"),
        ]

        # Add pixel spacing only if at least one lobe uses ellipse-based volume
        if needs_pixel_spacing and st.session_state.pixel_spacing is not None:
            param_list.append(('Pixel Spacing', f"{st.session_state.pixel_spacing[0]:.3f} mm"))

        # Add volume source info for each lobe
        left_vol_source = "Manual" if st.session_state.manual_volume_left is not None else "Ellipse"
        right_vol_source = "Manual" if st.session_state.manual_volume_right is not None else "Ellipse"
        param_list.append(('Right Volume Source', left_vol_source))
        param_list.append(('Left Volume Source', right_vol_source))

        param_list.extend([
            ('Total Volume', f"{total_volume_ml:.3f} mL"),
            ('Maximum Uptake', f"{max_uptake_percent:.2f}%"),
            ('Required Activity', f"{activity_mbq:.2f} MBq")
        ])

        param_data = {
            'Parameter': [p[0] for p in param_list],
            'Value': [p[1] for p in param_list]
        }

        df_params = pd.DataFrame(param_data)
        st.dataframe(df_params, use_container_width=True, hide_index=True)
        

        st.success(f"✓ Calculation complete! Required activity: **{activity_mbq:.2f} MBq**")

        # PDF Report Generation Section
        st.markdown("---")
        st.subheader("Generate PDF Report")

        # Check if patient info is available
        patient_info_available = all([
            st.session_state.patient_name is not None,
            st.session_state.patient_birth_date is not None,
            st.session_state.acquisition_dates is not None
        ])

        if not patient_info_available:
            st.warning("⚠️ Patient information is missing. Please go to the Thyroid Uptake Analysis section and upload DICOM files to extract patient data.")
            st.info("Alternatively, you can manually enter patient information below:")

        # Input fields for report
        col1, col2 = st.columns(2)

        with col1:
            physicist_name = st.text_input(
                "Physicist Name*",
                value="dott.",
                help="Name of the physicist performing the analysis"
            )

            patient_name_input = st.text_input(
                "Patient Name",
                value=st.session_state.patient_name if st.session_state.patient_name else "",
                help="Patient name (extracted from DICOM or manually entered)"
            )

        with col2:
            birth_date_input = st.text_input(
                "Birth Date (DD/MM/YYYY)",
                value=st.session_state.patient_birth_date if st.session_state.patient_birth_date else "",
                help="Patient birth date"
            )

            acquisition_dates_input = st.text_input(
                "Acquisition Dates (e.g., 05-06/10/2025)",
                value=st.session_state.acquisition_dates if st.session_state.acquisition_dates else "",
                help="Acquisition dates in format: first_day-last_day/month/year"
            )

        # Calculate activities for 120 Gy and 200 Gy
        activity_120 = (conversion_factor * 120.0 * total_volume_ml) / (max_uptake_decimal * activity_divisor)
        activity_200 = (conversion_factor * 200.0 * total_volume_ml) / (max_uptake_decimal * activity_divisor)

        st.info(f"The report will include activities for both 120 Gy ({activity_120:.1f} MBq) and 200 Gy ({activity_200:.1f} MBq)")

        # Generate PDF button
        if st.button("📄 Generate PDF Report", type="primary"):
            if not physicist_name:
                st.error("Please enter the physicist name")
            elif not patient_name_input or not birth_date_input or not acquisition_dates_input:
                st.error("Please fill in all patient information fields")
            else:
                try:
                    with st.spinner("Generating PDF report..."):
                        pdf_buffer = generate_pdf_report(
                            fisico=physicist_name,
                            nome=patient_name_input,
                            data_nascita=birth_date_input,
                            data_acquisizione=acquisition_dates_input,
                            captazione_4h=st.session_state.uptake_4h,
                            captazione_24h=st.session_state.uptake_24h,
                            volume=total_volume_ml,
                            attivita_120=activity_120,
                            attivita_200=activity_200
                        )

                        # Generate filename with patient name and date
                        filename = f"Report_{patient_name_input.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"

                        st.download_button(
                            label="💾 Click here to download the PDF",
                            data=pdf_buffer,
                            file_name=filename,
                            mime="application/pdf"
                        )

                        st.success("✓ PDF report generated successfully!")

                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display messages about missing data
    if not all_data_available:
        st.markdown("---")
        if not volume_data_available and not uptake_data_available:
            st.error("❌ No data available for dosimetry calculations. Please complete the following steps:")
            st.markdown("""
            1. Go to **Volume & Ellipse Fitting** mode and save both left and right volume data
            2. Go to **Thyroid Uptake Analysis** mode and save uptake data (choose either 4H or 24H phantom normalization)
            3. Return to this page to view dosimetry calculations
            """)
        elif not volume_data_available:
            st.warning("⚠️ Volume data is missing. To complete the activity calculation:")
            st.markdown("""
            - Go to **Volume & Ellipse Fitting** mode and either:
              - Save both left and right volume data from ellipse fitting, OR
              - Use the manual volume override option for difficult cases
            - Return to this page to view the complete dosimetry calculations
            """)
        elif not uptake_data_available:
            st.warning("⚠️ Uptake data is missing. To complete the activity calculation:")
            st.markdown("""
            - Go to **Thyroid Uptake Analysis** mode and save uptake data (choose either 4H or 24H phantom normalization)
            - Return to this page to view the complete dosimetry calculations
            """)
        
        if st.button("🔄 Clear All Saved Data"):
            st.session_state.volume_left = None
            st.session_state.volume_right = None
            st.session_state.uptake_4h = None
            st.session_state.uptake_24h = None
            st.session_state.pixel_spacing = None
            st.session_state.manual_volume_left = None
            st.session_state.manual_volume_right = None
            st.success("All saved data cleared!")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
This app combines three medical physics analysis workflows:

**Volume & Ellipse Fitting:**
- Analyzes DICOM volume images
- Splits into left/right regions
- Fits ellipses to each side
- Save volume parameters

**Thyroid Uptake:**
- Creates ROI masks
- Computes RawIntDen values
- Calculates radioactive decay corrections
- Computes thyroid uptake percentages
- Save uptake results

**Dosimetry Calculations:**
- Uses saved volume and uptake data
- Calculates total thyroid volume
- Determines required activity for treatment
- Converts pixel measurements to physical units

Developed for Medical Physics applications.
""")