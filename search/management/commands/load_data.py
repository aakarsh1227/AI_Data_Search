import pandas as pd
from django.core.management.base import BaseCommand
from search.models import CompanyInfo
from sentence_transformers import SentenceTransformer

class Command(BaseCommand):
    help = 'Load company data and generate vector embeddings'

    def handle(self, *args, **kwargs):
        # 1. Load the AI Model (Downloads automatically on first run)
        print("Loading Hugging Face model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 2. Read the CSV (Make sure data.csv is in your project root)
        print("Reading data.csv...")
        try:
            df = pd.read_csv('data.csv')
        except FileNotFoundError:
            print("Error: data.csv not found in project root.")
            return

        items_to_add = []
        
        # 3. Process each row
        print(f"Processing {len(df)} rows...")
        for _, row in df.iterrows():
            # Construct a rich descriptive sentence for the AI to understand
            text_representation = (
                f"Company: {row['Company Name']}. "
                f"Industry: {row['Industry']} ({row['Sector']}). "
                f"Located in {row['HQ State']}. "
                f"Annual Revenue: ${row['Annual Revenue 2022-2023 (USD in Billions)']} Billion. "
                f"Employees: {row['Employee Size']}."
            )

            # Generate Vector (The Magic Part)
            vector = model.encode(text_representation)

            items_to_add.append(CompanyInfo(
                name=row['Company Name'],
                content=text_representation,
                embedding=vector
            ))

        # 4. Save to DB
        CompanyInfo.objects.all().delete() # Clear old data
        CompanyInfo.objects.bulk_create(items_to_add)
        print(f"Success! {len(items_to_add)} companies loaded into the Vector DB.")