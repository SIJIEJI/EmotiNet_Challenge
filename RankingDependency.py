import requests
import time
import json
import os

# import AU and Emotion recognition algorithm
from AUrecognitionIR import image2AUvect

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
        request = dict()
        request['image_results'] = result
        response = requests.post(self.__build_url(self.URL['RESULT'], True), json=request)
        data = response.json()
        if len(data['submissionResult']) == 1:
            print(data['submissionResult'][0]['upload_response'])
            
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


dependency = RankingDependency('ekxsq0m0q7wk387x3jpo', 'q9ohr', '2')

# Specify track number. '1' for AU recognition. For ENC-2020, only track 1 is accepted. 
trackId = '1' 

foldersDict = dependency.get_folders_for_track(trackId)
folders = [s['folder_id'] for s in foldersDict]

folder_full = folders;

t = time.time()

# saving all uploaded folders in a json file 
upload_history_dir = 'uploaded_folder.json'

# getting the folders that haven't been uploaded yet
if os.path.isfile(upload_history_dir):
    with open(upload_history_dir,'r') as f_history_read:
        folders_uploaded = json.load(f_history_read)
    print(' '.join(folders_uploaded)+' already uploaded')
    folders_temp = list(set(folder_full) - set(folders_uploaded))
else:
    folders_uploaded = []
    folders_temp = folder_full

# uploading - main loop
for folder in folders_temp:
    result = []
    images = dependency.get_image_for_folder(folder)
    for index, image in enumerate(images):
        image_url = image['image_url']

        # retry when there is connection error 
        for retry_idx in range(1,100):
            try:
                image_data = requests.get(image_url)
                break
            except:
                print('Connection Error while uploading: '+image_url+'\n')
                time.sleep(0.5)
        # write the image to 'analysis_image.jpg'        
        with open('analysis_image' + '.jpg', 'wb') as f:
            f.write(image_data.content)
            f.close()
        #import pdb;pdb.set_trace()
        image_result = dict()
        image_result['image_result'] = image2AUvect('analysis_image.jpg')  # Algorithm Line
        print(index)
        image_result['image_id'] = image['image_id']
        image_result['folder_id'] = folder
        import pdb; pdb.set_trace()
        result.append(image_result)
        
    if len(result) != 0:
        dependency.send_evaluation_result(result)
        folders_uploaded.append(folder)
        print(folder+' completed.')
        with open(upload_history_dir,'w+') as f:
            json.dump(folders_uploaded,f)
            
elapses = time.time() - t
 


