from django.db import models

class PredResults(models.Model):

    Variance = models.FloatField()
    Entropy = models.FloatField()
    Skewness = models.FloatField()
    Kurtosis = models.FloatField()
    ASM = models.FloatField()
    Homogeneity = models.FloatField()
    Dissimilarity = models.FloatField()
    Correlation = models.FloatField()
    classification = models.CharField(max_length=30)

    def __str__(self):
        return self.classification