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
import  os, urllib
from .function import channel_intensity
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
        if not pic_urls:
            pic = request.data.get('pic')
            obj = Image(name = name, pic = pic)
            obj.save()
            im = settings.MEDIA_ROOT+'pics/'+pic.name
            R,G,B = channel_intensity(im)
            return Response({"message":"Intensito: R:"+str(R)+" G:" +str(G)+ " B:" + str(B)}, status.HTTP_200_OK)
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
            im = settings.MEDIA_ROOT+'pics/'+F.name
            R,G,B = channel_intensity(im)
            return Response({"message": "Intensità: R:"+str(R)+" G:"+str(G)+" B:"+str(B) }, status.HTTP_200_OK)

        

# ViewSets define the view behavior.
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = (IsAdminOrReadOnly, )