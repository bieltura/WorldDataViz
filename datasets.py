import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast

import graph


class BCNDataset(Dataset):
    def __init__(self):

        # Load the connexion dataset
        connection_dataset = pd.read_csv('custom_datasets/2020_connexions_barris_districtes.csv')

        # Get the variables from datasets: codes, names and connections
        self.codes_districts = np.array(connection_dataset.Codi_Districte)
        self.codes_neighborhood = np.array(connection_dataset.Codi_Barri)

        self.names_disctrics = np.array(connection_dataset.Nom_Districte)
        self.names_neighborhood = np.array(connection_dataset.Nom_Barri)

        self.num_districts = np.amax(np.array(connection_dataset.Codi_Districte))
        self.num_neighborhoods = np.amax(np.array(connection_dataset.Codi_Barri))

        # Add the district adjacency (remove duplicates bc of neighborhoods)
        districts_adjacency = []
        old_adjacency = ""
        for adjacency in np.array(connection_dataset.Connexions_Districte):
            if adjacency != old_adjacency:
                districts_adjacency.append(ast.literal_eval(adjacency))
                old_adjacency = adjacency

        # Add the neighborhood adjacency
        neighborhood_adjacency = []
        for adjacency in np.array(connection_dataset.Connexions_Barri):
            neighborhood_adjacency.append(ast.literal_eval(adjacency))

        # Generate the adjacency graph
        self.districts_graph = graph.get_adjacency_matrix(self.num_districts, districts_adjacency)
        self.neighborhoods_graph = graph.get_adjacency_matrix(self.num_neighborhoods, neighborhood_adjacency)

    def __len__(self):
        return self.num_neighborhoods

    def __getitem__(self, idx):

        # Select the district and neighborhood and weight its graph
        neighborhood = self.codes_neighborhood[idx]
        district = self.codes_districts[idx]

        neighborhood_graph = graph.weight_graph(self.neighborhoods_graph, neighborhood)
        district_graph = graph.weight_graph(self.districts_graph, district)

        sample = {'district_graph':     district_graph,
                  'district_name':      self.names_disctrics[idx],
                  'district':           district,
                  'neighborhood_graph': neighborhood_graph,
                  'neighborhood_name':  self.names_neighborhood[idx],
                  'neighborhood':       neighborhood}

        return sample


class OpenDataDataset(Dataset):
    def __init__(self, years, dataset_name, root_dir=''):

        # Load the graph datasets:
        self.graph_datasets = BCNDataset()

        self.dataset = []
        for year in years:
            self.dataset.append(pd.read_csv(str(root_dir) + '/' + str(year) + dataset_name))

        # Load all the districts and neighborhood for the dataset
        self.codes_districts = []
        self.codes_neighborhood = []
        self.labels = []
        
        for dataset in self.dataset:
            self.codes_districts.append(np.array(dataset.Codi_Districte))
            self.codes_neighborhood.append(np.array(dataset.Codi_Barri))

        self.codes_neighborhood = np.array(self.codes_neighborhood).flatten()
        self.codes_districts = np.array(self.codes_districts).flatten()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the weighted graph (only neighborhood is needed)
        neighborhood = self.codes_neighborhood[idx]
        sample_graph = self.graph_datasets[neighborhood-1]

        sample = {'district_graph':     sample_graph['district_graph'],
                  'district':           sample_graph['district'],
                  'neighborhood_graph': sample_graph['neighborhood_graph'],
                  'neighborhood':       sample_graph['neighborhood'],
                  'label':              self.labels[idx]}

        return sample


class LloguersDataset(OpenDataDataset):
    def __init__(self, years, lloguer_mensual=False):
        super().__init__(years, '_lloguer_preu_trim.csv', root_dir='opendata_bcn_datasets/lloguer_m2_mensual')

        self.title = "Precio Alquiler"
        if not lloguer_mensual:
            self.title = self.title + " (€/m2)"
        else:
            self.title = self.title + " (€/mes)"

        # Variable to separate type of rent
        tipus_lloguer = []

        # Load all the datasets for each year
        for dataset in self.dataset:
            self.labels.append(np.array(dataset.Preu))
            tipus_lloguer.append(np.array(dataset.Lloguer_mitja))

        # Concatenate the array into one single dataset
        self.labels = np.array(self.labels).flatten()
        tipus_lloguer = np.array(tipus_lloguer).flatten()
        
        # Filter if there is any NaN label
        removing_indices = []
        for (i, label) in enumerate(self.labels):
            if pd.isnull(label):
                removing_indices.append(i)

        # Now filter for the type of rent acording to boolean variable
        for (i, lloguer) in enumerate(tipus_lloguer):
            if not ((lloguer_mensual and len(lloguer) < 35) or (not lloguer_mensual and len(lloguer) > 35)):
                removing_indices.append(i)

        self.labels = np.delete(self.labels, removing_indices)
        self.codes_districts = np.delete(self.codes_districts, removing_indices)
        self.codes_neighborhood = np.delete(self.codes_neighborhood, removing_indices)

    def get_title(self):
        return self.title


class AtursDataset(OpenDataDataset):
    def __init__(self, years, specific_sexe=None, specific_months=[1,2,3,4,5,6,7,8,9,10,11,12]):
        super().__init__(years, '_atur_per_sexe.csv', root_dir='opendata_bcn_datasets/atur_per_sexe')

        self.title = "Número de aturados"
        if specific_sexe is not None:
            self.title = self.title + " (" + specific_sexe + ")"

        # Variable to separate type of rent
        sexes = []
        months = []

        # Load all the datasets for each year
        for dataset in self.dataset:
            self.labels.append(np.array(dataset.Nombre))
            sexes.append(np.array(dataset.Sexe))
            months.append(np.array(dataset.Mes))

        # Concatenate the array into one single dataset
        self.labels = np.hstack(self.labels)
        sexes = np.hstack(sexes)
        months = np.hstack(months)

        # Now filter for the type of rent acording to boolean variable
        removing_indices = []
        for (i, label) in enumerate(self.labels):
            if pd.isnull(label):
                removing_indices.append(i)
            if self.codes_districts[i] == 99:
                removing_indices.append(i)
    
        if specific_sexe is not None:
            for (i, sexe) in enumerate(sexes):
                if not sexe == specific_sexe:
                    removing_indices.append(i)
    
        if months is not None:
            for (i, month) in enumerate(months):
                if not month in specific_months:
                    removing_indices.append(i)

        self.labels = np.delete(self.labels, removing_indices)
        self.codes_districts = np.delete(self.codes_districts, removing_indices)
        self.codes_neighborhood = np.delete(self.codes_neighborhood, removing_indices)
    
    def get_title(self):
        return self.title

class EspVidaDataset(OpenDataDataset):
    def __init__(self):
        super().__init__([2017], '_est_salut_publica_esp_vida.csv', root_dir='opendata_bcn_datasets/esp_vida')

        self.title = "Esperanza de Vida"

        # Load all the datasets for each year
        for dataset in self.dataset:
            self.labels.append(np.array(dataset.Nombre))

        # Concatenate the array into one single dataset
        self.labels = np.hstack(self.labels)

        # Now filter for the type of rent acording to boolean variable
        removing_indices = []
        for (i, label) in enumerate(self.labels):
            if pd.isnull(label):
                removing_indices.append(i)
            if self.codes_districts[i] == 99:
                removing_indices.append(i)

        self.labels = np.delete(self.labels, removing_indices)
        self.codes_districts = np.delete(self.codes_districts, removing_indices)
        self.codes_neighborhood = np.delete(self.codes_neighborhood, removing_indices)
    
    def get_title(self):
        return self.title

class SuperficieHabitatge(OpenDataDataset):
    def __init__(self, years):
        super().__init__(years, '_loc_hab_sup_mitjana.csv', root_dir='opendata_bcn_datasets/superficie_mitjana_locals_habitatge')

        self.title = "Superfície media habitage"

        # Load all the datasets for each year
        for dataset in self.dataset:
            self.labels.append(np.array(dataset.Sup_mitjana_m2))

        # Concatenate the array into one single dataset
        self.labels = np.hstack(self.labels)

        # Now filter for the type of rent acording to boolean variable
        removing_indices = []
        for (i, label) in enumerate(self.labels):
            if pd.isnull(label):
                removing_indices.append(i)
            if self.codes_districts[i] == 99:
                removing_indices.append(i)

        self.labels = np.delete(self.labels, removing_indices)
        self.codes_districts = np.delete(self.codes_districts, removing_indices)
        self.codes_neighborhood = np.delete(self.codes_neighborhood, removing_indices)
    
    def get_title(self):
        return self.title

class EdatMitjana(OpenDataDataset):
    def __init__(self, years):
        super().__init__(years, '_loc_hab_edat_mitjana.csv', root_dir='opendata_bcn_datasets/edat_mitjana')

        self.title = "Edad media de la población"

        # Load all the datasets for each year
        for dataset in self.dataset:
            self.labels.append(np.array(dataset.Edat_mitjana))

        # Concatenate the array into one single dataset
        self.labels = np.hstack(self.labels)

        # Now filter for the type of rent acording to boolean variable
        removing_indices = []
        for (i, label) in enumerate(self.labels):
            if pd.isnull(label):
                removing_indices.append(i)
            if self.codes_districts[i] == 99:
                removing_indices.append(i)

        self.labels = np.delete(self.labels, removing_indices)
        self.codes_districts = np.delete(self.codes_districts, removing_indices)
        self.codes_neighborhood = np.delete(self.codes_neighborhood, removing_indices)
    
    def get_title(self):
        return self.title


class RendaTributaria(OpenDataDataset):
    def __init__(self, years):
        super().__init__(years, '_rendatributariamitjanaunitatconsum.csv', root_dir='opendata_bcn_datasets/renda_tributaria')

        self.title = "Renda Tributaria"

        # Load all the datasets for each year
        for dataset in self.dataset:
            self.labels.append(np.array(dataset.Import_Euros))

        # Concatenate the array into one single dataset
        self.labels = np.hstack(self.labels)

        # Now filter for the type of rent acording to boolean variable
        removing_indices = []
        for (i, label) in enumerate(self.labels):
            if pd.isnull(label):
                removing_indices.append(i)
            if self.codes_districts[i] == 99:
                removing_indices.append(i)

        self.labels = np.delete(self.labels, removing_indices)
        self.codes_districts = np.delete(self.codes_districts, removing_indices)
        self.codes_neighborhood = np.delete(self.codes_neighborhood, removing_indices)
    
    def get_title(self):
        return self.labels

class SuperficieLloguer(OpenDataDataset):
    def __init__(self, years):
        super().__init__(years, '_lloguer_sup_trim.csv', root_dir='opendata_bcn_datasets/sup_lloguer_m2')

        self.title = "Superficie (m2) de alquiler"

        # Load all the datasets for each year
        for dataset in self.dataset:
            self.labels.append(np.array(dataset.Nombre))

        # Concatenate the array into one single dataset
        self.labels = np.hstack(self.labels)

        # Now filter for the type of rent acording to boolean variable
        removing_indices = []
        for (i, label) in enumerate(self.labels):
            if pd.isnull(label):
                removing_indices.append(i)
            if self.codes_districts[i] == 99:
                removing_indices.append(i)

        self.labels = np.delete(self.labels, removing_indices)
        self.codes_districts = np.delete(self.codes_districts, removing_indices)
        self.codes_neighborhood = np.delete(self.codes_neighborhood, removing_indices)
    
    def get_title(self):
        return self.title