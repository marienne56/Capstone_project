import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import calendar
import streamlit as st

# Initialize Faker with English locale
fake = Faker('en_US')
@st.cache_data(ttl=3600) 
def generate_synthetic_data(num_records=5000, start_date=datetime(2023, 1, 1), 
                           end_date=datetime(2025, 12, 31), cities=None, client_types=None,
                           identifiers=None):
    """
    Generates synthetic water consumption data for prediction application.
    
    Args:
        num_records: Number of records to generate
        start_date: Start date for the data
        end_date: End date for the data
        cities: List of cities (generates random cities if None)
        client_types: List of client types (uses default types if None)
        identifiers: List of user identifiers (generates randomly if None)
        
    Returns:
        DataFrame with synthetic data
    """
    # Default values
    if cities is None:
        cities = ["Yamoussoukro", "San Pedro", "Jacqueville", "Abidjan", "Man", 
                 "Korhogo", "Séguela", "Katiola", "Boundiali", "Bouaké"]
    
    if client_types is None:
        client_types = ["Residential", "Commercial", "Industrial"]
    
    if identifiers is None:
        # Generate unique random identifiers
        identifiers = [f"USER_{i:04d}" for i in range(1, 101)]
    
    # Function to generate a random date within a range
    def random_date(start, end):
        delta = end - start
        random_days = random.randint(0, delta.days)
        return start + timedelta(days=random_days)
    
    # Base consumption by client type
    base_consumption = {
        "Residential": {"base": 100, "variation": 30},
        "Commercial": {"base": 350, "variation": 100},
        "Industrial": {"base": 1200, "variation": 300}
    }
    
    # Base consumption by city (multiplication factors)
    city_factors = {}
    for city in cities:
        # Each city has a base coefficient between 0.7 and 1.3
        city_factors[city] = random.uniform(0.7, 1.3)
    
    # Unit price by client type
    price_units = {
        "Residential": 0.15,
        "Commercial": 0.12,
        "Industrial": 0.10
    }
    
    # Data creation
    data = []
    invoice_id = 1
    
    for i in range(num_records):
        # Generate a random date within the range
        period_start = random_date(start_date, end_date)
        
        # Simulate a billing period (15 days to 2 months)
        duree_facturation = random.randint(15, 60)
        period_end = period_start + timedelta(days=duree_facturation)
        
        # Extract periods
        annee_debut = period_start.year
        mois_debut = period_start.month
        trimestre_debut = (period_start.month - 1) // 3 + 1
        semestre_debut = 1 if period_start.month <= 6 else 2
        
        # Generate city and client type
        ville = random.choice(cities)
        type_client = random.choice(client_types)
        
        # Generate identifier
        identifier = random.choice(identifiers)
        
        # Generate client name (consistent with identifier)
        # For simplicity, use the same name for each identifier
        if "USER_" in identifier:
            client_name = fake.name()
        else:
            client_name = identifier  # Use identifier as default name
        
        # Unit price based on client type
        prix_unitaire = price_units[type_client]
        
        # Consumption generation with business logic
        # Base consumption according to client type
        base = base_consumption[type_client]["base"]
        variation = base_consumption[type_client]["variation"]
        
        # Random variation around the base
        base_conso = base + random.uniform(-variation, variation)
        
        # City factor
        city_factor = city_factors[ville]
        
        # Seasonal factor
        seasonal_factor = 1.0
        
        # Higher consumption in summer for agriculture and residential (air conditioning)
        if mois_debut in [6, 7, 8]:
            if type_client == "Commercial":
                seasonal_factor = 1.2
            elif type_client == "Residential":
                seasonal_factor = 1.15
        
        # Higher consumption in winter for residential (heating)
        elif mois_debut in [12, 1, 2]:
            if type_client == "Residential":
                seasonal_factor = 1.2
        
        # Peak commercial activity during holidays
        elif mois_debut in [11, 12]:
            if type_client == "Commercial":
                seasonal_factor = 1.25
        
        # General variation by season
        month_factor = 1 + 0.1 * np.sin(2 * np.pi * (mois_debut - 3) / 12)
        
        # Upward trend factor over time
        years_from_start = annee_debut - start_date.year
        trend_factor = 1 + (0.02 * years_from_start)
        
        # Random variation to add noise
        noise_factor = random.uniform(0.85, 1.15)
        
        # Calculate consumption taking into account all factors
        daily_conso = base_conso * city_factor * seasonal_factor * month_factor * trend_factor * noise_factor / 30
        conso = daily_conso * duree_facturation
        
        # Round to 2 decimal places
        conso = round(conso, 2)
        
        # Calculate invoice amount
        mont_fact = round(conso * prix_unitaire, 2)
        
        # Calculate taxes (16.5%)
        taxes = round(mont_fact * 0.165, 2)
        
        # Generate payment status
        statut_paiement = random.choices(
            ["Paid", "Unpaid", "Late"],
            weights=[0.7, 0.2, 0.1],
            k=1
        )[0]
        
        # Generate penalty for late/unpaid payments
        penalite = 0.0
        if statut_paiement == "Late":
            penalite = 5.0
        elif statut_paiement == "Unpaid":
            penalite = 10.0
        
        # Calculate total amount
        mon_total = mont_fact + taxes + penalite
        
        # Generate a unique meter number for the identifier
        num_comp = f"COMP_{identifier.split('_')[1]}" if "_" in identifier else f"COMP_{random.randint(1000, 9999)}"
        
        # Meter type
        type_comp = random.choice(["Normal Meter", "Smart Meter"]) if type_client == "Residential" else "Industrial Meter"
        
        # Generate email
        email = f"{client_name.lower().replace(' ', '.')}@example.com"
        
        # Generate phone number
        num_tel = fake.phone_number()
        
        # Generate unique invoice ID
        id_facture = f"FACT{invoice_id:04d}"
        invoice_id += 1
        
        # Add to data list
        data.append({
            "id_consum": i + 1,
            "identifier": identifier,
            "id_facture": id_facture,
            "ClientName": client_name,
            "Ville": ville,
            "periodeDebut": period_start,
            "periodeFin": period_end,
            "Conso": conso,
            "PrixUnitaire": prix_unitaire,
            "Taxes": taxes,
            "StatutPaiement": statut_paiement,
            "penalite": penalite,
            "TypeClient": type_client,
            "NumComp": num_comp,
            "TypeComp": type_comp,
            "email": email,
            "NumTel": num_tel,
            "MontFact": mont_fact,
            "MonTotal": mon_total,
            "duree_facturation": duree_facturation
        })
    
    # Create DataFrame
    df_synthetic = pd.DataFrame(data)
    
    # Add some outliers to make the data more realistic
    outliers_count = int(num_records * 0.05)  # 5% outliers
    for _ in range(outliers_count):
        row_idx = random.randint(0, num_records - 1)
        # Multiply consumption by a factor between 2 and 5
        df_synthetic.at[row_idx, 'Conso'] *= random.uniform(2, 5)
        
        # Update associated amounts
        df_synthetic.at[row_idx, 'MontFact'] = round(df_synthetic.at[row_idx, 'Conso'] * df_synthetic.at[row_idx, 'PrixUnitaire'], 2)
        df_synthetic.at[row_idx, 'Taxes'] = round(df_synthetic.at[row_idx, 'MontFact'] * 0.165, 2)
        df_synthetic.at[row_idx, 'MonTotal'] = df_synthetic.at[row_idx, 'MontFact'] + df_synthetic.at[row_idx, 'Taxes'] + df_synthetic.at[row_idx, 'penalite']
    
    return df_synthetic