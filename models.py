from django.db import models
from django.utils import timezone
import random
import string


class CusUser(models.Model):
    username = models.CharField(max_length=100)
    first_name = models.CharField(max_length=100)
    surname = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=15)
    password = models.CharField(max_length=128)  # Increased length to accommodate hashed passwords
    class Meta:
        db_table = 'CusUser'

def generate_random_room_number():
    """Generate a random room number between 1 and 4."""
    return str(random.randint(1, 4))

class FaceImage(models.Model):
    username = models.CharField(max_length=100)
    image_data = models.FileField(upload_to='face_images/')
    room_number = models.CharField(max_length=10)
    timestamp = models.DateTimeField()

    def __str__(self):
        return f'{self.username} - {self.timestamp}'