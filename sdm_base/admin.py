from django.contrib import admin
from .models import *

# Register your models here.

class ChlldAdmin(admin.ModelAdmin):
	list_display = ('name', 'roll')

class AttributeAdmin(admin.ModelAdmin):
	list_display = ('name', 'roll', 'person_gender')

	def name(self, obj):
		return obj.child.name
	
	def roll(self, obj):
		return obj.child.roll
	
	def person_gender(self, obj):
		return 'Male' if obj.gender == 1 else 'Female'

admin.site.register(Child, ChlldAdmin)
admin.site.register(Attribute, AttributeAdmin)