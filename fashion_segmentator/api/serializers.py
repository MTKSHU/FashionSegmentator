from rest_framework import serializers
from .models import Image
from django.contrib.auth.models import User

class ImageSerializers (serializers.ModelSerializer):
    class Meta:
        model = Image
        fields = '__all__'

# Serializers define the API representation.
class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'is_staff')