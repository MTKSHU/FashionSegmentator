from django.db import models
from django.utils import timezone

class Image (models.Model):
    name = models.CharField(max_length=40)
    pic = models.ImageField(null=True,blank=True,upload_to=r"pics/")
    urls = models.CharField(null=True,blank=True,max_length=500)
    time = models.DateTimeField(editable=False)
    
    
    # Methods
    def save(self, *args, **kwargs):
        # On save, update timestamps
        self.time = timezone.now()
        return super(Image, self).save(*args, **kwargs)

    def __str__(self):
        return str(self.name + "; " + str(self.time))