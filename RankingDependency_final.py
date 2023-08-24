import requests
import time
import json
import os
import numpy as np
# import AU and Emotion recognition algorithm
# from AUrecognition import image2AUvect

class RankingDependency:
    AUTHENTICATION_TOKEN = ''
    USER_ID = '' # this is account id after login to the evaluation server
    RESULT_BATCH = ''

    DOMAIN = 'http://ranking-env-test.us-east-2.elasticbeanstalk.com/'
    URL = {
        'RESULT': '{}/api/{}/{}/{}/submit_results_folder',
        'FOLDER': '{}/api/{}/{}/getFoldersTrack',
        'IMAGES': '{}/api/{}/{}/getImages'
    }

    def __init__(self, authentication_token='0', user_id='0', result_batch='0'):
        if authentication_token == '0' and user_id == '0' and result_batch == '0':
            raise ValueError(
                'You must provide an Authentication Token and User Id and a Result Batch which is not zero.')

        self.AUTHENTICATION_TOKEN = authentication_token
        self.USER_ID = user_id
        self.RESULT_BATCH = result_batch

    def get_folders(self):
        response = requests.post(self.__build_url(self.URL['FOLDER'], False))
        result_dict = response.json()['folderName']
        error_list = response.json()['errorList']
        if len(error_list) == 0:
            return result_dict
        else:
            raise ValueError('Authentication Failure. Please check you authentication credentials.')

    def get_folders_for_track(self, track_id):
        response = requests.post(self.__build_url(self.URL['FOLDER'], False), json={'trackId': track_id})
        result_dict = response.json()['resultList']
        error_list = response.json()['errorList']
        if len(error_list) == 0:
            return result_dict
        else:
            raise ValueError('Authentication Failure. Please check you authentication credentials.')

    def get_image_for_folder(self, folder):
        response = requests.post(self.__build_url(self.URL['IMAGES'], False), json={'folder': folder})
        result_dict = response.json()['resultList']
        error_list = response.json()['errorList']
        if len(error_list) == 0:
            return result_dict
        else:
            raise ValueError('Authentication Failure. Please check you authentication credentials.')
    
    def send_evaluation_result(self, result):
        ns_per_part = 1000
        for idx_part in range(0,len(result), ns_per_part):
            part_result = result[idx_part:idx_part + ns_per_part]
            request = dict()
            request['image_results'] = part_result
            response = requests.post(self.__build_url(self.URL['RESULT'], True), json=request)
            # import pdb; pdb.set_trace()
            data = response.json()
            if len(data['submissionResult']) == 1:
                print(data['submissionResult'][0]['upload_response'])


    # def send_evaluation_result(self, result):
    #     request = dict()
    #     request['image_results'] = result
    #     response = requests.post(self.__build_url(self.URL['RESULT'], True), json=request)
    #     data = response.json()
    #     if len(data['submissionResult']) == 1:
    #         print(data['submissionResult'][0]['upload_response'])
            
    def __build_url(self, str, isReuslt):
        if isReuslt:
            return str.format(self.DOMAIN,
                              self.AUTHENTICATION_TOKEN,
                              self.USER_ID,
                              self.RESULT_BATCH)
        else:
            return str.format(self.DOMAIN,
                              self.AUTHENTICATION_TOKEN,
                              self.USER_ID)


dependency = RankingDependency('ekxsq0m0q7wk387x3jpo', 'q9ohr', '6')

# Specify track number. '1' for AU recognition. For ENC-2020, only track 1 is accepted. 
trackId = '1' 

foldersDict = dependency.get_folders_for_track(trackId)
folders = [s['folder_id'] for s in foldersDict]

folder_full = folders;

t = time.time()

# saving all uploaded folders in a json file 
upload_history_dir = 'uploaded_folder.json'

#['sa7jlr', 'wj6mrz', 'fymabs', 'af75tx', 'o91wbb', '0g4ic3', 'sz7dyl', '3tmoac', 'g7lclk', 'ae4q59', 'k2uy74', 'xosehg', 'ngj49d', 'v1089a', 'lgeaz7', 'w8efdn', 'lofcr8', 'c89qdy', '52knld', 'ex8iz8', 'mb83af', 'smvtg9', 'ff2asl', 'g2npap', 'rm33ne', 'cz5of2', 'vltf5n', '00dcb2', 'bzr791', 'c37rjd', '3w6q9b', '2fo5cm', '39r1nz', 'dmdoft', 'ag61uv', 'usjslj', 'bd9ej2', 'zapxnl', 'iozpep', 'v31jx0', 'wy9jjw', '37aqi3', '8smmw7', '5jakyn', 'a4ugw8', 'qv3mwd', '4hbgap', 'dwfzpe', 'hp6x8c', 'fprqgi', 'eeqry9', 'e6as75', 'p3bi1y', 'rdwtok', 'qmm60w', 'bq5xfu', 'muu8v1', '23ut7j', '9ypgix', 'k76alf', 'c1mdxa', '6d7f05', 'pghdxy', '21pypz', 'tco5wd', 'kzgi2u', '3pa7ew', '0w6lnb', '56pov9', 'n4xg26', 't7rj61', 'bmyrl0', 'qcfeh8', 'eie2lj', '3baql2', 'c1blyi', '95r7qz', 'o80bat', 'k6o0fv', 'v71s92', '8uta88', 'x564zh', 'lg7knq', '5mfchy', 'ssvrgh', '11hsep', '9hyyi6', 'cje56p', '14pvkj', '0ldw96', 'g98cta', 'j8s8tx', 'fhs31w', '4uh20t', 'm7fo0n', 'ie7t7j', 'loaj37', '5upqrc', 'msl3d6', 'wiu0fu', 'jn1er4', 'cgn94y', 'prphek', 'ywan2k', '42co44', '51ayos', 'zzen3e', '8x8k6m', 't3xxtf']

# getting the folders that haven't been uploaded yet
if os.path.isfile(upload_history_dir):
    #with open(upload_history_dir,'r') as f_history_read:
    with open(upload_history_dir,encoding='utf-8-sig', errors='ignore') as f_history_read:
        
        folders_uploaded = json.load(f_history_read)
    print(' '.join(folders_uploaded)+' already uploaded')
    folders_temp = list(set(folder_full) - set(folders_uploaded))
else:
    folders_uploaded = []
    folders_temp = folder_full


###### read the result#######

# results_dict = np.load('final_test_result.npy',allow_pickle='TRUE').item()
# results_modi = {}
# with open('final_test_imgs_align.txt', 'r') as imf:
#         for line in imf:
#             #import pdb; pdb.set_trace()
#             line = line.strip()
#             img_name,_ = line.split('.')
#             img_name_sub1,img_name_sub2 = img_name.split('_')
#             img_name_sub1 = img_name_sub1 + '_'
#             results_modi[str(img_name)] = results_dict[img_name_sub2]
#np.save('final_test_result_modi.npy', results_modi)
results_modi = np.load('final_test_result_modi.npy',allow_pickle='TRUE').item()

#import pdb; pdb.set_trace()
#rename_dict = result_dict
# import pdb; pdb.set_trace()
# uploading - main loop
for folder in folders_temp:
    result = []
    images = dependency.get_image_for_folder(folder)
    import time
    #import pdb; pdb.set_trace()
    # pause for 2 second for testing on latency
    time.sleep(0.5)
    for index, image in enumerate(images):
        #import pdb; pdb.set_trace()
        # image_url = image['image_url']

        # # retry when there is connection error 
        # for retry_idx in range(1,100):
        #     try:
        #         image_data = requests.get(image_url)
        #         break
        #     except:
        #         print('Connection Error while uploading: '+image_url+'\n')
        #         time.sleep(0.5)
        # # write the image to 'analysis_image.jpg'        
        # with open('analysis_image' + '.jpg', 'wb') as f:
        #     f.write(image_data.content)
        #     f.close()
        # import pdb;pdb.set_trace()
        image_result = dict()
        # error_img = []
        f_error = open("err_test_img.txt","w+")
        #mport pdb; pdb.set_trace()
        # image_result['image_result'] = image2AUvect('analysis_image.jpg')  # Algorithm Line
        #方案1：image_result['image_result'] = image2AUvect('img_path' + image['image_id'] +'.jpg')
        
        
        # except (Exception, requests.exceptions.ProxyError) as e:
        #     print(e)
        if image['image_id'] in results_modi:
            print("image_id:",image['image_id'])
            image_result['image_result'] = results_modi[image['image_id']]
            image_result['image_id'] = image['image_id']
            image_result['folder_id'] = folder
            result.append(image_result)
        else:
            #image_result['image_result'] = None
            image_result['image_id'] = image['image_id']
            image_result['folder_id'] = folder
            # image_result['image_result'] = None
            # result.append(image['image_id'])
            f_error.write(image_result['folder_id'] + '/' + image['image_id']+ '\n')  
        #import pdb; pdb.set_trace()
        #{'image_id': 'image_y8339r', 'folder_id': 'sa7jlr', 'image_result': [0, 0, 999, 0, 0, 0, 999, 999, 0, 0, 999, 0, 999, 999, 0, 999, 0, 0, 999, 0, 999, 999, 999, 1, 0, 0, 999, 0, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 0, 999, 999, 999, 999, 999, 999, 999, 0, 0, 0, 0, 0, 0, 999, 999, 999, 999]}
            
    # import pdb; pdb.set_trace()

    if len(result) != 0:
        # import pdb; pdb.set_trace()
        dependency.send_evaluation_result(result)
        folders_uploaded.append(folder)
        print(folder+' completed.')
        with open(upload_history_dir,'w+') as f:
            json.dump(folders_uploaded,f)
            
elapses = time.time() - t
 


