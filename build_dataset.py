import os
import itertools
from PIL import Image
from sklearn.model_selection import train_test_split

'''
Before run the script make sure that you have the correct file structure, as given below, to avoid path errors.
These folders must exist in advance, as of this version of the code.

-Project folder (name won't matter here)
  - code/build_dataset.py

- Dataset
  - cezanne2photo
  - vangogh2photo
  - monet2photo
  - ukiyoe2photo
  - ArtNet
    - train
    - test
    - validation
    
** Note that this script will create ethe ArtNet from both image restoration and style transfer learning. To only keep the art restoration
comment out the parts of code of which are sectioned as `B`
'''
parent_path = '../../../Dataset/'
cezanne_path = os.path.join(parent_path, 'cezanne2photo')
vangogh_path = os.path.join(parent_path, 'vangogh2photo')
monet_path = os.path.join(parent_path, 'monet2photo')
ukiyoe_path = os.path.join(parent_path, 'ukiyoe2photo')
db_store_path = os.path.join(parent_path, 'ArtNet')


def build_dataset(art_paths, store_path, sets, split_validation):
  for dbset in sets:

    set_store_path = os.path.join(store_path, dbset)

    # Train or Test case
    prev_files = [f for f in os.listdir(set_store_path)]
    if len(prev_files) != 0:
      for f in prev_files:
        os.remove(os.path.join(set_store_path, f))

    # validation case
    if dbset == 'train' and split_validation:
      prev_files = [f for f in os.listdir(os.path.join(store_path, 'validation'))]
      if len(prev_files) != 0:
        for f in prev_files:
          os.remove(os.path.join(os.path.join(store_path, 'validation'), f))

    for art_path in art_paths:

      set_path_A, set_path_B = os.path.join(art_path, '{}A'.format(dbset)), os.path.join(art_path, '{}B'.format(dbset))
      set_path_A_dir = os.listdir(set_path_A)
      set_path_B_dir = os.listdir(set_path_B)

      set_path_A_dir_valid = []
      set_path_B_dir_valid = []
      counter_index = 0

      if dbset == 'train' and split_validation:
        valid_counter_index = 0
        # same as concatenating the two sets A,B and took the 5% for validation. This however is more representative of both sets.
        set_path_A_dir, set_path_A_dir_valid = train_test_split(set_path_A_dir, test_size=0.05, random_state=141)

        set_path_B_dir, set_path_B_dir_valid = train_test_split(set_path_B_dir, test_size=0.05, random_state=141)

      for image_path, valid_image_path in itertools.zip_longest(set_path_A_dir + set_path_B_dir, set_path_A_dir_valid + set_path_B_dir_valid,
                                                                fillvalue=''):

        # validation case
        if valid_image_path != '':
          valid_img_art = Image.open(os.path.join(set_path_A if len(valid_image_path.split('-')) == 1 else set_path_B, valid_image_path))
          valid_filename = art_path.split('/')[-1].split('2')[0] + '_{}{}.jpg'.format('A' if len(valid_image_path.split('-')) == 1 else 'B',
                                                                                      valid_counter_index)

        img_art = Image.open(os.path.join(set_path_A if len(image_path.split('-')) == 1 else set_path_B, image_path))
        filename = art_path.split('/')[-1].split('2')[0] + '_{}{}.jpg'.format('A' if len(image_path.split('-')) == 1 else 'B', counter_index)

        # validation case
        if valid_image_path != '':
          print("Writing file {} into {} dataset".format(valid_filename, 'validation'))
          valid_img_art.save(os.path.join(os.path.join(store_path, 'validation'), valid_filename))
          valid_counter_index += 1

        print("Writing file {} into {} dataset".format(filename, dbset))
        img_art.save(os.path.join(set_store_path, filename))
        counter_index += 1


if __name__ == '__main__':
  art_paths = [cezanne_path, vangogh_path, monet_path, ukiyoe_path]
  build_dataset(art_paths, db_store_path, sets=['train', 'test'], split_validation=True)
