# facedetection/admin.py

from django.contrib import admin
from django.utils.safestring import mark_safe
from .models import CusUser, FaceImage

class UserAdmin(admin.ModelAdmin):
    list_display = ('username', 'first_name', 'surname', 'email', 'phone')
    search_fields = ('username', 'email')

class FaceImageAdmin(admin.ModelAdmin):
    list_display = ('username', 'image_tag', 'room_number', 'timestamp')
    search_fields = ('username',)
    list_filter = ('room_number', 'timestamp')

    def image_tag(self, obj):
        if obj.image_data:
            return mark_safe('<img src="{url}" width="150" height="150" />'.format(url=obj.image_data.url))
        else:
            return "No Image"

    image_tag.short_description = 'Image'  # Sets the column name in the admin list display

admin.site.register(CusUser, UserAdmin)
admin.site.register(FaceImage, FaceImageAdmin)
