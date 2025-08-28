from .GQA.gqa import GQADataset, GQACollator
from .WhatsUp.dataset_zoo import *
from .VSR.vsr import VSRDataset, VSRCollator
from .POPE.pope import POPEDataset, POPECollator
from .COCO.coco import COCODataset, COCOCollator
from .VQA.vqa import VQADataset, VQACollator
from .MME.mme import MMEDataset, MMECollator
from .TextVQA.textvqa import TextVQADataset, TextVQACollator
from .SQA.sqa import SQADataset, SQACollator
from .MMBench.mmb import MMBDataset, MMBCollator
from .custom_data import GQASquareImages