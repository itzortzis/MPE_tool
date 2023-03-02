# MPE_tool
Mammogram Patches Extraction tool

<img src="https://user-images.githubusercontent.com/105294556/222407074-0efadf82-9342-4700-abbe-a7421f393bd5.svg" width="200" height="200">

This tool can be used for the extraction of patches from three open
and well known datasets (INbreast, mini-MIAS and CBIS_DDSM) containing digital or digitized mammograms.

Required Libraries:

     random, cv2, matplotlib, PIL, numpy, pydicom, anot_core

**Note:** The anot_core library can be installed using the following command:

pip3 install git+https://github.com/itzortzis/MPE_tool.git


Files and description:
----------------------
```
MPE_tool
│    README.md
│
└─── MPE_core
│    │   mpe.py  #The main file containing the ImgPatchExtractor class
│   
│   
└─── notebook
     │   mpe.ipynb #Colab/Jupyter notebook version of the tool
```


## Installation
----------------

The MPE_tool can be cloned from here or it can be installed using Python pip tool

- Option 1: Clone the repository and see the demo files in python and notebook folders
- Option 2:
  - Install tool using ```pip3 install git+https://github.com/itzortzis/MPE_tool.git```
  - Import the needed components ```from MPE_core import mpe```

Enjoy!!!


INBreast dataset:
-----------------
Inês C. Moreira, Igor Amaral, Inês Domingues, António Cardoso, Maria João Cardoso, Jaime S. Cardoso,
INbreast: Toward a Full-field Digital Mammographic Database,
Academic Radiology,
Volume 19, Issue 2,
2012,
Pages 236-248,
ISSN 1076-6332,
https://doi.org/10.1016/j.acra.2011.09.014.
(https://www.sciencedirect.com/science/article/pii/S107663321100451X)


mini-MIAS dataset:
------------------
J Suckling et al (1994),
MINI-MIAS: The Mammographic Image Analysis Society Digital Mammogram Database Exerpta Medica, International Congress,
Series 1069,
pp375-378.
http://peipa.essex.ac.uk/info/mias.html


CBIS_DDSM:
----------
Lee, R. S., Gimenez, F., Hoogi, A., Miyake, K. K., Gorovoy, M., & Rubin, D. L. (2017),
CBIS_DDSM: A curated mammography data set for use in computer-aided detection and diagnosis research,
In Scientific Data (Vol. 4, Issue 1),
Springer Science and Business Media LLC,
https://doi.org/10.1038/sdata.2017.177
