import cv2
import PIL
import random
import numpy as np
import pydicom as pdcm
from skimage import exposure
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt
from anot_core import annotation as anot
from skimage.exposure import adjust_sigmoid
from skimage.filters import difference_of_gaussians, sobel, sobel_h, sobel_v


class ImgPatchExtractor:

  def __init__(self, **kwargs):
    
    self.dn         = kwargs["dn"] # Dataset name
    self.p_size     = kwargs["p_size"]
    self.h_patches  = kwargs["hppi"]
    self.nh_patches = kwargs["nhppi"]
    self.mar_thresh = kwargs["mar"] #0.01 # Mass area ratio threshold
    self.bar_thresh = kwargs["bar"] #0.8  # Breast area ratio threshold
    self.p_idx      = 0
    self.info       = kwargs["info"] # This is used only for the MIAS dataset

    self.img_path   = kwargs["img_path"]
    self.mask_path  = kwargs["mask_path"]
    self.root_dir   = kwargs["root_dir"]
    self.img_proc   = self.exec_pipeline()


  # Exec_pipeline:
  # --------------
  # This is the orchestrating function that combines the individual
  # mechanisms of this class in order to achieve the extraction of patches
  # from any input image. It returns True if everything goes as planned. 
  # Otherwise, it returns a False value.
  def exec_pipeline(self):
    if self.dn == "CBIS":
      self.correct_paths()

    self.np_img = self.load_img_obj(self.img_path, self.dn)
    self.np_mask = self.load_mask(self.mask_path, self.dn)

    temp_img, brdrs = self.crop_img(self.np_img)

    self.step_init   = (np.min(self.np_img.shape)) // 200
    self.step        = self.step_init
    self.valid_pair  = self.np_img.shape == self.np_mask.shape
    
    if not self.valid_pair:
      return False

    if self.dn == "CBIS" or self.dn == "INBreast":
      self.np_img, self.np_mask = self.flip_img_hor(temp_img, temp_img, self.np_mask[brdrs[2]:brdrs[3], brdrs[0]:brdrs[1]])
    else:
      self.np_img, self.np_mask = self.flip_img_hor(temp_img, self.np_img, self.np_mask)
      self.np_img = self.np_img[brdrs[2]:brdrs[3], brdrs[0]:brdrs[1]]
      self.np_mask = self.np_mask[brdrs[2]:brdrs[3], brdrs[0]:brdrs[1]]

    e = np.where(self.np_img < 0.1)

    self.raw_img = self.np_img.copy()
    if self.dn == "INBreast":
      # print("xaxaxa")
      self.apply_filter()
      self.np_img[e] = 0
      self.np_mask = np.where(self.np_mask > 0, 1, 0)
    self.filt_img = np.zeros((self.raw_img.shape[0], self.raw_img.shape[1], 4))
    img = self.raw_img.copy()
    # print("Img before: ", np.count_nonzero(self.np_img))
    self.extract_features(img)
    # print("Img after: ", np.count_nonzero(self.np_img))

    self.patches = self.extract_patches(True)
      
    return True


  def apply_filter(self):
    self.np_img = exposure.equalize_hist(self.np_img)
    
  def histo(self, img):
    return exposure.equalize_hist(img)

  def sigmo(self, img):
    return adjust_sigmoid(img, cutoff=0.5, gain=10, inv=False)
  
  def entro(self, img):
    return entropy(img, disk(10))
  

  def extract_features(self, img):
    img = img/4095
    self.filt_img[:, :, 0] = img
    self.filt_img[:, :, 1] = self.histo(img)
    self.filt_img[:, :, 2] = self.sigmo(img)
    self.filt_img[:, :, 3] = self.entro(img)
    # self.filt_img[:, :, 4] = self.histo(self.raw_img)
    return
    

  # Crop_img:
  # ---------
  # Given an img object (numpy array), this function performs a simple cropping
  # in order to get rid of any non-essential regions containing no breast tissue.
  #
  # --> img: pixel array of the image (numpy array)
  # <-- crpd: the cropped pixel array - image with no black colored padding
  def crop_img(self, img):
      borders = np.nonzero(img)
      t = np.min(borders[0])
      b = np.max(borders[0])
      l = np.min(borders[1])
      r = np.max(borders[1])

      crpd = img[t:b, l:r]

      return crpd, (l, r, t, b)


  # Correct_paths:
  # --------------
  # Given the absolute paths to the image and the corresponding annotation
  # mask, this function replaces the prefix with the the one specified by the
  # root_dir class property. This function is essential for the CBIS_DDSM.

  def correct_paths(self):
    self.img_path  = self.img_path.replace('CBIS-DDSM/jpeg', self.root_dir)
    self.mask_path = self.mask_path.replace('CBIS-DDSM/jpeg', self.root_dir)


  # Load_img_obj:
  # -------------
  # Given the image path and the name of the dataset where it belongs to, this 
  # function utilizes the proper function to load the image object.
  #
  # --> img_path: the path to the original image
  # --> dn: dataset name: INBREAST, CBIS or MIAS
  def load_img_obj(self, img_path, dn):
    assert dn == 'CBIS' or dn == 'MIAS' or dn == 'INBreast', "Wrong dataset name"

    if dn == "CBIS":
      return self.load_CBIS_DDSM_img(img_path)
    elif dn == "MIAS":
      return self.load_MIAS_img(img_path)
    else:
      return self.load_INBreast_img(img_path)


  # Load_CBIS_DDSM_img:
  # -------------------
  # This function implements the custom solution for loading the CBIS_DDSM
  # images.
  # 
  # --> img_path: path to the original image
  # <-- g_img: the imported pixel array (numpy array)
  def load_CBIS_DDSM_img(self, img_path):
    img   = PIL.Image.open(img_path)
    g_img = img.convert("L")

    return self.img2array(g_img)
  

  # Load_MIAS_img:
  # -------------------
  # This function implements the custom solution for loading the MIAS
  # images.
  # 
  # --> img_path: path to the original image
  def load_MIAS_img(self, img_path):
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


  # Load_INBreast_img:
  # -------------------
  # This function implements the custom solution for loading the INBreast
  # images.
  # 
  # --> img_path: path to the original image
  # <-- arr: the imported pixel array (numpy array)
  def load_INBreast_img(self, img_path):
    img   = pdcm.dcmread(self.root_dir + 'DICOM/' + img_path + '.dcm')
    arr   = img.pixel_array

    return arr


  # Load_mask_obj:
  # -------------
  # Given the image path and the name of the dataset where it belongs to, this 
  # function utilizes the proper function to load the annotation mask object.
  #
  # --> mask_path: the path to the annotation mask
  # --> dn: dataset name: INBREAST, CBIS or MIAS
  def load_mask(self, mask_path, dn):
    assert dn == 'CBIS' or dn == 'MIAS' or dn == 'INBreast', "Wrong dataset name"

    if dn == "CBIS":
      return self.load_CBIS_DDSM_mask(mask_path)
    elif dn == "MIAS":
      return self.load_MIAS_mask(mask_path)
    else:
      return self.load_INBreast_mask(mask_path)


  # Load_MIAS_mask:
  # -------------------
  # This function implements the custom solution for loading the MIAS
  # annotation mask.
  # 
  # <-- mask: the imported pixel array (numpy array)
  def load_MIAS_mask(self):
    mask = np.zeros((self.np_img.shape[0], self.np_img.shape[1]))
    print(self.info)
    center_coordinates = (int(self.info[4]), int(self.info[5]))
    radius = int(self.info[6])
    color = (255, 0, 0)
    thickness = -1
    mask = cv2.circle(mask, center_coordinates, radius, color, thickness)
    mask = np.flipud(mask)

    return mask


  # Load_CBIS_DDSM_mask:
  # -------------------
  # This function implements the custom solution for loading the CBIS_DDSM
  # annotation masks.
  # 
  # --> mask_path: path to the annotation mask
  # <-- mask: the imported pixel array (numpy array)
  def load_CBIS_DDSM_mask(self, mask_path):
    mask   = PIL.Image.open(mask_path)
    mask = mask.convert("L")

    return self.img2array(mask)

  
  # Load_INBreast_mask:
  # -------------------
  # This function implements the custom solution for loading the INBreast
  # annotation masks.
  # 
  # --> mask_path: path to the annotation mask
  # <-- mask: the imported pixel array (numpy array)
  def load_INBreast_mask(self, img_path):
    xml = self.root_dir + "XML/"
    mask_obj = anot.Annotation(xml, self.mask_path, self.np_img.shape)
    mask = mask_obj.mask[:, :, 0]
    
    return mask


  # Img2array:
  # ----------
  # This function takes as input and image object and converts it to
  # numpy array
  # 
  # --> im: the image object
  def img2array(self, im):
    return np.asarray(im)


  # Is_healthy:
  # -----------
  # Given the annotation mask of a mammogram, this function returns True if
  # there is any non zero pixel.
  # 
  # --> gt: the annotation mask (numpy array)
  def is_healthy(self, gt):
    return np.sum(gt) == 0


  # P_contains_tissue:
  # ------------------
  # Given a mammogram patch, this function checks its validity according to
  # the area of breast tissue compared with the total patch area. The function
  # returns True if the breast area is beyond the predefined threshold.
  # 
  # --> p: self.p_size x self.psize mammogram patch
  def p_contains_tissue(self, p):
    p_area  = p.shape[0] ** 2
    b_area  = np.count_nonzero(p)
    b_cover = b_area / p_area

    return b_cover > self.bar_thresh


  # Is_mar_valid:
  # ------------------
  # Given a mammogram patch, this function checks its validity according to
  # the area of the mass compared with the total patch area. The function
  # returns True if the mass area is beyond the predefined threshold.
  # 
  # --> p: self.p_size x self.psize mammogram patch
  # --> gt: the corresponding annotation mask
  def is_mar_valid(self, p, gt):
    p_area  = p.shape[0] ** 2
    m_area  = np.count_nonzero(gt)
    m_cover = m_area / p_area

    return m_cover >= self.mar_thresh

  
  # Mass localization:
  # ------------------
  # Given the annotation mask, this function is looking for the borders
  # of the existing mass. Then, it calculates a specific region where the 
  # to left corner of the new random patch will be located. 
  # 
  # --> gt: the annotation mask (numpy array)
  # <-- min_h: the top border of the region
  # <-- max_h: the bottom border of the region
  # <-- min_w: the left border of the region
  # <-- max_w: the right border of the region
  def mass_localization(self, gt):
    borders = np.nonzero(gt)
    min_h = np.min(borders[0])
    max_h = np.max(borders[0])
    min_w = np.min(borders[1])
    max_w = np.max(borders[1])
    
    max_d = max(max_h - min_h, max_w - min_w)
    offset = int(0.1 * max_d)
    min_h = min_h - offset
    if min_h < 0:
      min_h = 0
    max_h = max_h - offset
    if max_h < 0:
      max_h = 0
    min_w = min_w - offset
    if min_w < 0:
      min_w = 0
    max_w = max_w - offset
    if max_w < 0:
      max_w = 0

    return (min_h, max_h, min_w, max_w)

  
  # Adapt_step:
  # -----------
  # According to the inputs, this function adapts the step that corresponds 
  # to the parsing technique used to extract patches from the original 
  # mammogram.
  # 
  # --> inc: if True the step is increased, otherwhise it is decreased
  # --> sw: if True, the step is not altered.
  def adapt_step(self, inc, sw):
    if not sw:
      return

    if inc:
      self.step += self.step_init * 2
    else:
      self.step -= self.step_init * 2
    
    if self.step < self.step_init:
      self.step = self.step_init

    if self.step > self.step_init + 100:
      self.step = self.step_init + 100


  # Flip_img_hor:
  # -------------
  # In an effort to achieve homogeneity, this function tries to flip
  # horizontally all the mammograms with "wrong" orientation.
  # 
  # --> test_region: essential part of the image
  # --> img: the img to be flipped
  # --> gt: the annotation mask to be flipped
  # <-- img: the flipped image
  # <-- gt: the flipped annotation mask
  def flip_img_hor(self, test_region, img, gt):

    lhs_col = np.sum(test_region[:, :200])
    rhs_col = np.sum(test_region[:, test_region.shape[1]-200 : test_region.shape[1]])

    if rhs_col > lhs_col:
      return np.fliplr(img), np.fliplr(gt)
    
    return img, gt


  # Find_p_chords:
  # --------------
  # This function finds random coordinations of the upper left corner of the
  # patch to be retrieved. These coordinations should be in a region specified
  # by the location of the mass.
  # 
  # --> borders: borders of the region where the upper left corner of the 
  #              patch is located.
  # <-- h: x-axis coordinate of the upper left corner of the patch to be 
  #        selected.
  # <-- w: y-axis coordinate of the upper left corner of the patch to be 
  #        selected.
  def find_p_coords(self, borders):
    height = self.np_img.shape[0] - self.p_size
    if height < 0:
      height = 0
    width  = self.np_img.shape[1] - self.p_size
    if width < 0:
      width = 0

    if self.h_patches > 0:
      h = random.randint(0, height)
      w = random.randint(0, width)
    else:
      h = random.randint(borders[0], borders[1])
      if h > height:
        h = random.randint(self.np_img.shape[0] - self.p_size - 100, self.np_img.shape[0] - self.p_size - 1)
      w = random.randint(borders[2], borders[3])
      if w > width:
        w = random.randint(self.np_img.shape[0] - self.p_size - 100, self.np_img.shape[0] - self.p_size - 1)

    return h, w

  
  # Patch_is_valid:
  # ---------------
  # Given a patch with the corresponding annotation mask, this function
  # ensures whether the patch is valid or not, according to the breast tissue
  # cover, the mass area, etc.
  # 
  # --> patch: numpy array with the patch information
  # --> gt: the annotation mask of the patch
  def patch_is_valid(self, patch, gt):
    if not self.p_contains_tissue(patch):
      return False
    if self.is_healthy(gt):
      self.adapt_step(False, True)
      if self.h_patches <= 0:
        return False
      self.h_patches -= 1
    else:
      if self.nh_patches <= 0 or not self.is_mar_valid(patch, gt):
        return False
      self.adapt_step(True, True)
      self.nh_patches -= 1

    return True
    

  # Extract_patches:
  # ----------------
  # This is the main function for the extraction of the patches from the 
  # current mammogram.
  # 
  # --> sw: enables/disables the adaptation of the step property
  # <-- extracted_patches: numpy array containing the retrieved patches
  def extract_patches(self, sw):
    shp = (self.h_patches + self.nh_patches, self.p_size, self.p_size, 5)
    extracted_patches = np.zeros(shp)
    self.p_idx = 0
    h = 0
    borders = self.mass_localization(self.np_mask)
    tries = 0
    while (self.h_patches > 0 or self.nh_patches > 0) and tries < 200:
      tries += 1
      h, w  = self.find_p_coords(borders)
      patch_filt = self.filt_img[h : h + self.p_size, w : w + self.p_size, :]
      patch = self.np_img[h : h + self.p_size, w : w + self.p_size]
      gt    = self.np_mask[h : h + self.p_size, w : w + self.p_size]
      
      if not self.patch_is_valid(patch, gt):
        continue
      # print("zaaaa")

      extracted_patches[self.p_idx, :, :, :4] = patch_filt
      extracted_patches[self.p_idx, :, :, 4] = gt

      self.p_idx += 1
    
    if tries > 198:
      print("Exceed tries ", self.p_idx)

    return extracted_patches
