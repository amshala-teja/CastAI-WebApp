def create SolarClient(title, pub_date):
    return SolarClient.objects.create(title = title, pub_date = pub_date) 
     
