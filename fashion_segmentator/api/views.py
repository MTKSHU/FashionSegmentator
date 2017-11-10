# coding=utf-8

from django.shortcuts import render
from rest_framework import viewsets, status
from .models import Image
from .serializers import ImageSerializers,UserSerializer
from django.contrib.auth.models import User
from .permissions import IsAdminOrReadOnly, AllowAny
from django.core.files.images import ImageFile
from django.core.exceptions import ValidationError
from rest_framework.decorators import detail_route
from rest_framework.response import Response
from django.http import HttpResponse
import  os, urllib, zipfile
from .function import channel_intensity
from .inference import predict
from django.conf import settings

class ImageView(viewsets.ModelViewSet):

    queryset = Image.objects.all()
    serializer_class = ImageSerializers
    permission_classes = (AllowAny, )
    def create(self, request):
        # Validate the incoming input (provided through post parameters)    
        name = request.data.get('name')
        pic  = request.data.get('pic')            
        pic_urls = request.data.get('urls')
        res_zip = request.data.get('zip_result')
        if not res_zip:
            res_zip = False;
            
        print(res_zip)
        if not pic_urls:
            pic = request.data.get('pic')
            obj = Image(name = name, pic = pic,zip_result=res_zip)
            obj.save()
            im_path = settings.MEDIA_ROOT+'pics/'+pic.name
            num_classes = 25
            model_weights = settings.WEIGHTS_ROOT + 'model.ckpt-1600' 
            save_dir = './output/'
            mask_file, my_json = predict(im_path,num_classes,model_weights,save_dir)
            
            if res_zip:
                # Create response zip file
                zipf = zipfile.ZipFile('result_data.zip', 'w', zipfile.ZIP_DEFLATED)
                zipf.write(save_dir + 'mask.png')
                zipf.write(save_dir + 'json_data.json')
                zipf.close()

                zipf_tDownload = open("result_data.zip",'r')

                response = HttpResponse(zipf_tDownload, content_type="application/zip")
                response['Content-Disposition'] = 'attachment; filename="result_data.zip"'
                return response
            else:
                return HttpResponse(my_json,content_type="application/json")
        else:
            try:
                out_path = os.path.join(os.path.dirname(__file__), 'img.jpg')
                urllib.request.urlretrieve(pic_urls, out_path)
                out_file = open(out_path, 'rb+')
                F = ImageFile(out_file)
                F.name = os.path.basename(pic_urls)
            except:
                raise ValidationError("Impossibile scaricare correttamente l'immmagine dal web.")
            obj = Image(name = name, urls = pic_urls, pic = F)
            obj.save()
            aim_path = settings.MEDIA_ROOT+'pics/'+F.name
            num_classes = 25
            model_weights = settings.WEIGHTS_ROOT + 'model.ckpt-1600'
            save_dir = './output/'
            mask_file, my_json = predict(im_path,num_classes,model_weights,save_dir,res_zip)
            
            if res_zip:
                # Create response zip file
                zipf = zipfile.ZipFile('result_data.zip', 'w', zipfile.ZIP_DEFLATED)
                zipf.write(save_dir + 'mask.png')
                zipf.write(save_dir + 'json_data.json')
                zipf.close()

                zipf_tDownload = open("result_data.zip",'r')

                response = HttpResponse(zipf_tDownload, content_type="application/zip")
                response['Content-Disposition'] = 'attachment; filename="result_data.zip"'
                return response
            else:
                return HttpResponse(my_json,content_type="application/json")
        

# ViewSets define the view behavior.
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (IsAdminOrReadOnly, )