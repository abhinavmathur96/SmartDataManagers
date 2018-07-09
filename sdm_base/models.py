from django.db import models

# Create your models here.
class Child(models.Model):
	name = models.CharField(max_length=50, blank=False)
	roll = models.CharField(max_length=10, unique=True, blank=False, primary_key=True)
	friends = models.ManyToManyField("self", blank=True, symmetrical=False)
	timestamp = models.DateTimeField(auto_now_add=True)

class Attribute(models.Model):
	child = models.OneToOneField(Child, on_delete=models.CASCADE, primary_key=True)
	friendliness = models.PositiveSmallIntegerField()
	sportiness = models.PositiveSmallIntegerField()
	kindness = models.PositiveSmallIntegerField()
	talkativeness = models.PositiveSmallIntegerField()
	extroversion = models.PositiveSmallIntegerField()
	popularity = models.PositiveSmallIntegerField()
	gender = models.PositiveSmallIntegerField()
	hardworking = models.PositiveSmallIntegerField()
	intelligence = models.PositiveSmallIntegerField()
	punctuality = models.PositiveSmallIntegerField()
	creativity = models.PositiveSmallIntegerField()
	leadership = models.PositiveSmallIntegerField()
	discipline = models.PositiveSmallIntegerField()
	loyalty = models.PositiveSmallIntegerField()

	class Meta:
		db_table='sdm_base_attributes'
