import os
import json
import torch
from momaapi import MOMA
from momaapi.lookup import Lookup
from momaapi.taxonomy import Taxonomy
from torch.utils.data import Dataset

class BBox2ActivityDataset(Dataset):
    def __init__(self, split='train', dir_moma=None, min_boxNum=10):
        # make a list of MOMA sact ids, where each sact has >= 10 bboxes
        if dir_moma is None or dir_moma == 'None':
            self.dir_moma = "/Users/akashaj/Work/Datasets/MOMA-LRG"
        elif dir_moma == "cs30":
            self.dir_moma = "/local-scratch/localhome/aabdujyo/datasets/MOMA-LRG"
        else:
            self.dir_moma = dir_moma

        self.moma = MOMA(self.dir_moma)
        self.taxonomy = Taxonomy(self.dir_moma)
        self.lookup = Lookup(self.dir_moma, self.taxonomy, reset_cache=False)
        with open(os.path.join(self.dir_moma,'anns','anns.json'), 'r') as json_file:
            self.video_data = json.load(json_file)

        self.ids_sact = self.moma.get_ids_sact(split=split)#[:200]
        
        # Get class list
        class_ids = self.moma.get_cids('sact', 0, 'train') # total 91 classes
        self.sact_class_names = self.moma.get_cnames(cids_sact=class_ids)
        actor_class_ids = self.moma.get_cids('actor', 0, 'train') # total 26 classes
        self.actor_class_names = self.moma.get_cnames(cids_actor=actor_class_ids)
        object_class_ids = self.moma.get_cids('object', 0, 'train') # total 227 classes
        self.object_class_names = self.moma.get_cnames(cids_object=object_class_ids)

        print(f'{split} set has {len(self.ids_sact)} samples')
        self.ids_sact_selected = self.select_ids_sact(min_boxNum)#[:100]
        print(f'Selected {len(self.ids_sact_selected)} samples that have >= {min_boxNum} bboxes')

    def select_ids_sact(self, min_boxNum):
        '''
        Select sact ids that have >= 10 bboxes
        '''
        ids_sact_and_boxNum = []

        for id_sact in self.ids_sact:
            ids_hoi = self.moma.get_ids_hoi(ids_sact=[id_sact])
            anns_hoi = self.moma.get_anns_hoi(ids_hoi=ids_hoi)
            
            boxNum = 0
            for ann_hoi in anns_hoi:
                boxNum += len(ann_hoi.actors) + len(ann_hoi.objects)

            ids_sact_and_boxNum.append((id_sact, boxNum))
        
        return [id_sact for (id_sact, boxNum) in ids_sact_and_boxNum if boxNum >= min_boxNum]
    
    def __len__(self):
        return len(self.ids_sact_selected)

    def __getitem__(self, idx):
        # Get the sact id from the list,
        # and get the corresponding bboxes across frames, activity labels
        # Bbox: [frame_num(one-hot), x, y, width, height, class(one-hot)]
        #       (1142, 289) 
        #        1142 = Max number of bounding boxes per subactivity
        #        289 = 32   + 4 +   (26 + 227)
        # Activity label: [activity class(one-hot)]  (91,1)
        id_sact = self.ids_sact_selected[idx]

        sact_video_fname = self.moma.get_paths(ids_sact=[id_sact])[0].split('/')[-1]
        ann_sact = self.moma.get_anns_sact(ids_sact=[id_sact])[0]
        ids_hoi = self.moma.get_ids_hoi(ids_sact=[id_sact])
        anns_hoi = self.moma.get_anns_hoi(ids_hoi=ids_hoi)

        sact_label_idx = torch.tensor([self.sact_class_names.index(ann_sact.cname)]) # total 91 classes
        # sact_label = torch.zeros(91, 1)
        # sact_label[sact_label_idx] = 1

        bbox_set = torch.zeros(1142, 289)
        bbox_mask = torch.ones(1142)

        count_bbox = 0
        # Normalize boxes based on video frame height
        # Min video frame height is 360 (max width is 720) 
        # (aspect ratio varies from 1.0 to 2.35)
        id_act = self.lookup.map_id(id_sact=id_sact, kind='id_act')
        act = list(filter(lambda x: x['file_name'].split('.')[0]==id_act, self.video_data))
        scaling_factor = 360 / act[0]['height']

        for idx, ann_hoi in enumerate(anns_hoi):  
            for actor in ann_hoi.actors:
                bbox_set[count_bbox, idx] = 1 # frame number      
                actor_label_idx = self.actor_class_names.index(actor.cname) # total 26 classes
                bbox_set[count_bbox, actor_label_idx+36] = 1 # 32 + 4 = 36
                bbox_set[count_bbox, 32] = actor.bbox.x * scaling_factor
                bbox_set[count_bbox, 33] = actor.bbox.y * scaling_factor
                bbox_set[count_bbox, 34] = actor.bbox.width * scaling_factor
                bbox_set[count_bbox, 35] = actor.bbox.height * scaling_factor

                count_bbox += 1

            for obj in ann_hoi.objects:
                bbox_set[count_bbox, idx] = 1 # frame number      
                obj_label_idx = self.object_class_names.index(obj.cname) # total 227 classes
                bbox_set[count_bbox, obj_label_idx+62] = 1 # 32 + 4 + 26 = 62
                bbox_set[count_bbox, 32] = obj.bbox.x * scaling_factor
                bbox_set[count_bbox, 33] = obj.bbox.y * scaling_factor
                bbox_set[count_bbox, 34] = obj.bbox.width * scaling_factor
                bbox_set[count_bbox, 35] = obj.bbox.height * scaling_factor

                count_bbox += 1

        bbox_mask[:count_bbox] = 0
        # decoder_tgt = torch.LongTensor([0])

        return bbox_set, bbox_mask.to(torch.bool), sact_label_idx, sact_video_fname#, decoder_tgt
