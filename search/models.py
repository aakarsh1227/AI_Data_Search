from django.db import models
from pgvector.django import VectorField

class CompanyInfo(models.Model):
    name = models.CharField(max_length=255)
    # This stores the readable text we want to search against
    content = models.TextField()
    # This stores the AI representation (384 dimensions for all-MiniLM-L6-v2)
    embedding = VectorField(dimensions=384)

    def __str__(self):
        return self.name