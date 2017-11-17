# coding=utf-8

from django.shortcuts import render
from rest_framework import viewsets, status, views
from .models import Image
from .serializers import ImageSerializers,UserSerializer
from django.contrib.auth.models import User
from .permissions import IsAdminOrReadOnly, AllowAny
from django.core.files.images import ImageFile
from django.core.exceptions import ValidationError
from rest_framework.decorators import detail_route
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
import  os, urllib, zipfile, requests, shutil
from .function import channel_intensity
from .inference import predict
from django.conf import settings
from rest_framework.renderers import JSONRenderer

class ImageView(viewsets.ModelViewSet):

    queryset = Image.objects.all()
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
            
        if not pic_urls:
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
                print("Stampa ZIP")
                response = HttpResponse(zipf_tDownload, content_type="application/zip")
                response['Content-Disposition'] = 'attachment; filename="result_data.zip"'
                return response
            else:
                print("Stampa JSON")
                return  HttpResponse(my_json,content_type="application/json")
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
            print(F.name)
            obj = Image(name = name, urls = pic_urls, pic = F,zip_result=res_zip)
            obj.save()
            im_path = settings.MEDIA_ROOT+'pics/'+F.name
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
                return  HttpResponse(my_json,content_type="application/json")

# ViewSets define the view behavior.
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (IsAdminOrReadOnly, )
