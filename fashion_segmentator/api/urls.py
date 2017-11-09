from django.conf.urls import url, include
from rest_framework import routers, serializers, viewsets
from .views import ImageView,UserViewSet




# Routers provide an easy way of automatically determining the URL conf.
router = routers.DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'images',ImageView)
router.register(r'',UserViewSet)
# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = router.urls;
