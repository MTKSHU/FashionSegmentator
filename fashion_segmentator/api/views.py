# coding=utf-8

from django.shortcuts import render
from rest_framework import viewsets, status, views
from .models import ImageUploaded
from .serializers import ImageSerializers,UserSerializer
from django.contrib.auth.models import User
from .permissions import IsAdminOrReadOnly, AllowAny
from django.core.files.images import ImageFile
from django.core.exceptions import ValidationError
from rest_framework.decorators import detail_route
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
import  os, urllib, zipfile, requests, shutil, json
from .function import channel_intensity
from .inference import predict
from django.conf import settings
from rest_framework.renderers import JSONRenderer
from resizeimage import resizeimage
from PIL import Image
from deeplab_resnet.utils import load_graph
import tensorflow as tf 
import numpy as np
import time,shutil

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


print('Loading the segmentation trained model')
x, y, y_score = load_graph(settings.WEIGHTS_ROOT + 'graph.pb')
print('Starting Session')
persistent_sess = tf.Session()

class ImageView(viewsets.ModelViewSet):

    queryset = ImageUploaded.objects.all()
    serializer_class = ImageSerializers
    permission_classes = (AllowAny, )
    def create(self, request):
        # Validate the incoming input (provided through post parameters)   
        serializer = ImageSerializers(data=request.data, context={'request': request})
        serializer.is_valid(raise_exception=True)
        name = serializer.validated_data.get('name')
        pic  = serializer.validated_data.get('pic')            
        pic_urls = serializer.validated_data.get('urls')
        res_zip = serializer.validated_data.get('zip_result')
        heavy = serializer.validated_data.get('heavy')

        
        
        print(heavy)
        if not pic_urls:
            im_path = settings.MEDIA_ROOT+'pics/'+pic.name
            if os.path.exists(im_path):
                print("Già caricato!")
                os.remove(im_path)
            else:
                print("Non ancora caricato!")

            obj = ImageUploaded(name = name, pic = pic,zip_result=res_zip,heavy=heavy)
            obj.save()
            num_classes = 25
            model_weights = settings.WEIGHTS_ROOT + 'model.ckpt-1600' 
            save_dir = './output/'
            scale = 1
            with open(im_path, 'r+b') as f:
                with Image.open(f) as image:
                    original_size = image.size
                    print(original_size)
                    if image.size[1] > 600:
                        scale = image.size[1]/600.0
                        cover = resizeimage.resize_height(image,600)
                        cover.save(im_path,image.format)
            im = Image.open(im_path)
            t1 = time.time()
            preds, scores = persistent_sess.run([y,y_score],feed_dict={x:np.array(im)})
            mask_file, my_json = predict(preds,scores,im_path,scale,num_classes,save_dir,heavy,t1)
            if res_zip:
                # Create response zip file
                my_json['original_size'] =  original_size
                with open(save_dir+'json_data.json', 'w') as outfile:
                    json.dump(my_json, outfile)
                zipf = zipfile.ZipFile('result_data.zip', 'w', zipfile.ZIP_DEFLATED)
                zipf.write(save_dir + 'mask.png')
                zipf.write(save_dir + 'json_data.json')
                zipf.close()
                zipf_tDownload = open("result_data.zip",'r')
                response = HttpResponse(zipf_tDownload, content_type="application/zip")
                response['Content-Disposition'] = 'attachment; filename="result_data.zip"'
                return response
            else:
                my_json['original_size'] =  original_size
                json_data = json.dumps(my_json)
                return  HttpResponse(json_data,content_type="application/json")
        else:
            try:
                print(pic_urls)
                r = requests.get(pic_urls, stream=True)
                if r.status_code == 200:
                    with open(os.path.join(os.path.dirname(__file__), 'img.jpg'), 'wb+') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
                        out_file = open(os.path.join(os.path.dirname(__file__), 'img.jpg'), 'rb+') 
                        F = ImageFile(out_file)
                        F.name = os.path.basename(pic_urls)
            except:
                raise ValidationError("Impossibile scaricare correttamente l'immmagine dal web.")
            im_path = settings.MEDIA_ROOT+'pics/'+F.name
            if os.path.exists(im_path):
                print("Già caricato!")
                os.remove(im_path)
            else:
                print("Non ancora caricato!")
            
            obj = ImageUploaded(name = name, urls = pic_urls, pic = F,zip_result=res_zip,heavy=heavy)
            obj.save()
            num_classes = 25
            model_weights = settings.WEIGHTS_ROOT + 'model.ckpt-1600' 
            save_dir = './output/'
            scale = 1
            with open(im_path, 'r+b') as f:
                with Image.open(f) as image:
                    original_size = image.size
                    if image.size[1] > 600:
                        scale = image.size[1]/600.0
                        cover = resizeimage.resize_height(image,600)
                        cover.save(im_path,image.format)
            im = Image.open(im_path)
            t1 = time.time()            
            preds, scores = persistent_sess.run([y,y_score],feed_dict={x:np.array(im)})
            mask_file, my_json = predict(preds,scores,im_path,scale,num_classes,save_dir,heavy,t1)
            if res_zip:
                # Create response zip file
                my_json['original_size'] =  original_size
                with open(save_dir+'json_data.json', 'w') as outfile:
                    json.dump(my_json, outfile)
                zipf = zipfile.ZipFile('result_data.zip', 'w', zipfile.ZIP_DEFLATED)
                zipf.write(save_dir + 'mask.png')
                zipf.write(save_dir + 'json_data.json')
                zipf.close()

                zipf_tDownload = open("result_data.zip",'r')

                response = HttpResponse(zipf_tDownload, content_type="application/zip")
                response['Content-Disposition'] = 'attachment; filename="result_data.zip"'
                return response
            else:
                my_json['original_size'] =  original_size
                json_data = json.dumps(my_json)
                return  HttpResponse(json_data,content_type="application/json")

# ViewSets define the view behavior.
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (IsAdminOrReadOnly, )
