import os
import cv2
import shutil
from sklearn.model_selection import train_test_split


def main():
    src_dir = "../../Inclass/catdog"
    dst_dir = "./catdog"
    
    all_filenames = os.listdir(src_dir)
    
    train_list, test_list = train_test_split(
                                    all_filenames,
                                    train_size=0.70,
                                    random_state=42)
    
    def prepare_images(filelist, data_type):
        print("Preparing", data_type)
        os.makedirs(os.path.join(dst_dir, data_type + "A"),
                    exist_ok=True)
        os.makedirs(os.path.join(dst_dir, data_type + "B"),
                    exist_ok=True)
        
        index = 0
        for filename in filelist:
            #shutil.copyfile()
            image = cv2.imread(os.path.join(src_dir, filename))
            image = cv2.resize(image, dsize=(256,256))
        
            if "dog" in filename:
                output_dir = data_type + "A"
            else:
                output_dir = data_type + "B"
            
            cv2.imwrite(os.path.join(dst_dir, 
                                     output_dir,
                                     filename),
                        image) 
            index += 1
            if index % 500 == 0 or index == len(filelist):
                print("Copied image", index, "of", len(filelist))
                
    prepare_images(train_list, "train")
    prepare_images(test_list, "test")

if __name__ == "__main__":
    main()