import numpy as np
from tensorflow.keras.utils import Sequence, load_img, img_to_array
import json
from sklearn.cluster import KMeans
## @brief Datagenerator for custim Yolov3. 
## @details This Datagenerator can be used for big data which would overload the RAM.  

class DataGenerator(Sequence):
    def __init__(self, path_main, list_id, batch_size, shuffle=True, img_shape=(416,416), data_augmentation=True, anchor=[0]):        
        ##
        # @brief This constructor initalizes the DataGenerator object.
        # @param path_main             The main path which includes the preprocessed data.
        # @param list_id               The list of the subject IDs.                 
        # @param batch_size            The size of each data batch.
        # @param shuffle               Whether to shuffle the data after each epoch. Default is True.
        # @param img_shape             Shape of the input images. Default is (416, 416)
        # @param data_augmentation     Whether to do data augmentation. Default is True.
        # @param anchor                The anchor which are used. Need to be a list with 3 normalized anchors. If Default anchors will be calculated. Default is [0].
        ##
        
        self.path_main = path_main
        self.list_id = list_id
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_shape = img_shape
        self.data_augmentation = data_augmentation
        self.anchor = anchor
        
        print("Loading DataGenerator")
        self.x, self.y = self._load_data()
        
        if len(self.anchor)==1:
            self._make_anchor()
            
        if self.data_augmentation==True:
            self._data_augmentation()
            
        print("Convert to grid")  
        self.y = [self._obj_cent2grid_cent(np.array(grid), anchor=anchor) for grid, anchor in zip([[13,13], [26,26], [52,52]], self.anchor)]    
        
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()

        

    def __len__(self):
        ##
        # @brief This method count the total number of batches.
        # @return Total number of batches.
        ##     
        return len(self.x)//self.batch_size
    
    def __getitem__(self, idx):
        ##
        # @brief       This method returns a batch of data.
        # @param idx   Index of batch
        # @return      A batch of data
        ##
        batch_idx = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        x = self.x[batch_idx]
        y = [self.y[i][batch_idx] for i in range(3)]
                
        return x, [y[0], y[1], y[2]]


    def get_anchor(self):
        ##
        # @brief This method returns the anchors which will be used.
        # @return A list with 3 anchors
        ##
        return self.anchor

    def on_epoch_end(self):
        ##
        # @brief This method will be called after each epoch and shuffle the data if it is indicated.
        ##
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def _make_anchor(self):
        ##
        # @brief This method create 3 anchor of the data with kmeans clustering.
        ##
        print("Generating Anchor")
        all_w, all_h = np.array([]), np.array([])

        for data in self.y:
            for label in data:
                if np.all(label!=0):
                    all_w = np.append(all_w, label[2])
                    all_h = np.append(all_h, label[3])
                
        x = np.swapaxes(np.array([all_w, all_h]), 0, 1)      

        kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit_predict(x)

        idx0 = np.where(kmeans==0)
        idx1 = np.where(kmeans==1)
        idx2 = np.where(kmeans==2)

        cluster0 = x[idx0]
        cluster1 = x[idx1]
        cluster2 = x[idx2]
        all_cluster = [cluster0, cluster1, cluster2]

        anchor = []
        for cluster in all_cluster:
             anchor.append(np.array([round(np.mean(cluster[:,0]), 3), round(np.mean(cluster[:,1]), 3)]))
    
        
        idx_anchors = np.argsort([np.mean(x) for x in anchor])
        idx_anchors_mod = [idx_anchors[2], idx_anchors[1], idx_anchors[0]]
        self.anchor = np.array([anchor[idx] for idx in idx_anchors_mod])

        
    
    def _load_image_pixels(self, filename):
        ##
        # @brief              This method returns one preprocessed image and its size.
        # @param filename     ID of the target image
        ##
        image = load_img(filename)
        width, height = image.size
        image = load_img(filename, target_size=self.img_shape)
        image = img_to_array(image)
        image = image.astype('float32')
        image /= 255.0
        image = np.expand_dims(image, 0)
        return image, width, height
    
    def _transform_bbox(self, box, image_h, image_w): 
        ##
        # @brief            This method transform coordinates of the target corners in bounding box coordinates.
        # @param box        Coordinates of target corners as dictionary and following keys: xmin, xmax, ymin, ymax
        # @param image_h    Original image height
        # @param image_w    Original image width
        ##
        x_min, x_max = box['xmin']/image_w, box['xmax']/image_w 
        y_min, y_max = box['ymin']/image_h, box['ymax']/image_h
        
        bw = x_max-x_min
        bh = y_max-y_min
        bx = x_min+bw/2
        by = y_min+bh/2
        
        return bx, by, bh, bw

    def _load_data(self):
        ##
        # @brief This method loads data of the next subject.
        ## 
        f = open(self.path_main+"kavsir_bboxes.json")
        data = json.load(f)
        
        max_obj = 2
        for idx, _id in enumerate(self.list_id):
            temp = data.get(_id[:-4])
            bbox = temp.get('bbox')
            if len(bbox) > max_obj:
                max_obj = len(bbox)

        images = np.zeros((len(self.list_id), self.img_shape[0], self.img_shape[1], 3))
        gt = np.zeros((len(self.list_id), max_obj, 4))

        for idx, _id in enumerate(self.list_id):    
            #images[idx] = preprocess_input(temp_img, net_h, net_w)
            images[idx], image_w, image_h = self._load_image_pixels(self.path_main+"images/"+_id)
            temp = data.get(_id[:-4])
            bbox = temp.get('bbox')

            label = np.zeros((max_obj, 4))
            
            
            for num, box in enumerate(bbox):
                bx, by, bh, bw = self._transform_bbox(box, image_h, image_w)
                label[num, 0] = bx
                label[num, 1] = by
                label[num, 2] = bw
                label[num, 3] = bh
            
            gt[idx] = label
        
        return images, gt
    
    def _rot90(self, x0, y0):
        ##
        # @brief This method rotates target image and label 90 degree clockwise.
        ## 
        def _rot_func(x, y):
            x_rel = x - 208
            y_rel = y - 208
            rotated_x_rel = y_rel
            rotated_y_rel = -x_rel
            rotated_x = rotated_x_rel + 208
            rotated_y = rotated_y_rel + 208

            return rotated_x, rotated_y 
        x1 = np.rot90(x0, k=3, axes=(1,2))
        y1 = np.zeros(y0.shape)
        
        for idx, data in enumerate(y0[0]):
            if data[0]==0: 
                continue
            x_coord, y_coord = _rot_func(data[0]*416, data[1]*416)
            y1[0,idx] = np.array([x_coord/416, y_coord/416, data[3], data[2]])
        return np.float32(x1), np.float32(y1)
        
        
    def _fliph(self, x0, y0):
        ##
        # @brief This method mirrors target image and label horitontally.
        ## 
        x1 = np.fliplr(x0)
        y1 = np.copy(y0)
        for idx_i, i in enumerate(y1[..., 0]):
            for idx_j, j in enumerate(i):
                if j==0: continue
                y1[idx_i, idx_j, 0] = 1-j        
        return np.float32(x1), np.float32(y1)
    
    
    def _flipv(self, x0, y0):
        ##
        # @brief This method mirrors target image and label vertically.
        ## 
        x1 = np.flipud(x0)  
        y1 = np.copy(y0)
        for idx_i, i in enumerate(y1[..., 1]):
            for idx_j, j in enumerate(i):
                if j==0: continue
                y1[idx_i, idx_j, 1] = 1-j
        return np.float32(x1), np.float32(y1)
    
    
    def _data_augmentation(self):
        ##
        # @brief This method do the data augmentation.
        ## 
        print("Do Data-Augmentation")
        x90, y90 = self._rot90(self.x, self.y)
        x180, y180 = self._rot90(x90, y90)
        x270, y270 = self._rot90(x180, y180)

        x0_fliph, y0_fliph = self._fliph(self.x, self.y)
        #x0_flipv, y0_flipv = self._flipv(self.x, self.y)
        
        x90_fliph, y90_fliph = self._fliph(x90, y90)
        #x90_flipv, y90_flipv = self._flipv(x90, y90)
        
        x180_fliph, y180_fliph = self._fliph(x180, y180)
        #x180_flipv, y180_flipv = self._flipv(x180, y180)
        
        x270_fliph, y270_fliph = self._fliph(x270, y270)
        #x270_flipv, y270_flipv = self._flipv(x270, y270)
        
        self.x = np.concatenate([np.float32(self.x), x90, x180, x270, 
                                x0_fliph, x90_fliph, x180_fliph, x270_fliph])
                                #x0_flipv, x90_flipv, x270_flipv])
        
        self.y = np.concatenate([np.float32(self.y), y90, y180, y270, 
                                y0_fliph, y90_fliph, y180_fliph, y270_fliph])
                                #y0_flipv, y90_flipv, y270_flipv])
                                
     

    def _obj_cent2grid_cent(self, grid_size, anchor=False):
        ##
        # @brief This method extract the object centered data into grid centered data.
        ## 
        def _inverse_sigmoid(y):
            return np.log(y / (1 - y))
        obj_cent = self.y
        grid_cent = np.zeros((obj_cent.shape[0], grid_size[0], grid_size[1], 5))
        # Convert each bounding box annotation to grid-centric format
        for sample_idx in range(obj_cent.shape[0]):
            for obj_idx in range(obj_cent.shape[1]):
                obj_annotation = obj_cent[sample_idx, obj_idx]
        
                # Skip if the object annotation is empty (e.g., all zeros)
                if np.all(obj_annotation == 0):
                    continue
        
                # Extract the object's class, center coordinates, width, and height
                x_center, y_center, width, height = obj_annotation
                
                # Calculate the grid cell indices
                grid_x = int(x_center * grid_size[0])
                grid_y = int(y_center * grid_size[1])
                
                temp_x = x_center - grid_x/grid_size[0]
                temp_y = y_center - grid_y/grid_size[1]
                if temp_x<0.0000001:
                    temp_x=0.0000001
                if temp_y<0.0000001:
                    temp_y=0.0000001
                if temp_x>0.9999999:
                    temp_x=0.9999999 
                if temp_y>0.9999999:
                    temp_y=0.9999999
                    
                tx = _inverse_sigmoid(temp_x) 
                ty = _inverse_sigmoid(temp_y)
                tw = np.log(width/anchor[0])
                th = np.log(height/anchor[1])
                
                # Update the corresponding grid cell with object information
                grid_cent[sample_idx, grid_y, grid_x, :] = [tx, ty, tw, th, 1]
                
        return grid_cent
        
        
        
        
        
        
        