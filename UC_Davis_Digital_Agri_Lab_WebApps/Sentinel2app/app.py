import os
import sys
import time
import io
from pathlib import Path


# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import ee
# Add Google Service account credential. Authenticates to the Earth Engine servers.
ee.Authenticate()


import math
from typing import Optional
from shiny import App, render, ui, reactive, Inputs, Outputs, Session, req
import ipyleaflet as L
from ipywidgets import Layout
from htmltools import css
import numpy as np
import pandas as pd
# from PIL import Image
from shinywidgets import output_widget, reactive_read, register_widget
from geopy.geocoders import Nominatim
import json
import requests
import traceback
from datetime import datetime, date
from typing import List
from utils import print_with_line_number
from timezonefinder import TimezoneFinder

# Library for ANN model loading
import tensorflow
import joblib

# Sentinel 2 Bands
# Sentinel-2 carries the Multispectral Imager (MSI). This sensor delivers 13 spectral bands ranging from 10 to 60-meter pixel size.

# Its blue (B2), green (B3), red (B4), and near-infrared (B8) channels have a 10-meter resolution.
# Next, its red edge (B5), near-infrared NIR (B6, B7, and B8A), and short-wave infrared SWIR (B11 and B12) have a ground sampling distance of 20 meters.
# Finally, its coastal aerosol (B1) and cirrus band (B10) have a 60-meter pixel size.
# Band	Resolution	Central Wavelength	Description
# B1	60 m	443 nm	Ultra Blue (Coastal and Aerosol)
# B2	10 m	490 nm	Blue
# B3	10 m	560 nm	Green
# B4	10 m	665 nm	Red
# B5	20 m	705 nm	Visible and Near Infrared (VNIR)
# B6	20 m	740 nm	Visible and Near Infrared (VNIR)
# B7	20 m	783 nm	Visible and Near Infrared (VNIR)
# B8	10 m	842 nm	Visible and Near Infrared (VNIR)
# B8a	20 m	865 nm	Visible and Near Infrared (VNIR)
# B9	60 m	940 nm	Short Wave Infrared (SWIR)
# B10	60 m	1375 nm	Short Wave Infrared (SWIR) - excluded
# B11	20 m	1610 nm	Short Wave Infrared (SWIR)
# B12	20 m	2190 nm	Short Wave Infrared (SWIR)

tf = TimezoneFinder()


# You can use different URLs to load remote sensing image data from various sources
# In this example, we use image data from Google Earth Engine
GEEurl = 'https://earthengine.googleapis.com/v1alpha/projects/earthengine-legacy/maps/{mapid}/tiles/{z}/{x}/{y}?token={token}'
GEEmap_id = ''  # Replace with your Google Earth Engine Map ID
GEEtoken = ''  # Replace with your Google Earth Engine Token

# Custom Loss Function with Covariance Penalty
def custom_loss(lam, cov_real_data):
    def loss(y_true, y_pred):
        mse_loss = tensorflow.reduce_mean(tensorflow.square(y_true - y_pred))
        cov_pred = tensorflow.linalg.matmul(tensorflow.transpose(y_pred - tensorflow.reduce_mean(y_pred, axis=0)), 
                                    (y_pred - tensorflow.reduce_mean(y_pred, axis=0))) / tensorflow.cast(tensorflow.shape(y_pred)[0], tensorflow.float32)
        cov_penalty = tensorflow.reduce_sum(tensorflow.square(cov_pred - cov_real_data))
        return mse_loss + lam * cov_penalty
    return loss

def load_model_and_preprocessors(model_path, cov_real_data_path, scaler_X_path, scaler_y_path):
    # Load the covariance matrix
    cov_real_data = np.load(cov_real_data_path)
    # Load the trained model
    model = tensorflow.keras.models.load_model(model_path, custom_objects={'loss': custom_loss(1e-6, cov_real_data)})
    # Load the input data scaler
    scaler_X = joblib.load(scaler_X_path)
    # Load the output data scaler
    scaler_Y = joblib.load(scaler_y_path)
    return model, scaler_X, scaler_Y

# Load ANN model and preprocessors
model, loaded_scaler_X, loaded_scaler_Y = load_model_and_preprocessors(
    r"ANN_assests\model",
    r"ANN_assests\cov_real_data.npy",
    r"ANN_assests\scaler_X.pkl",
    r"ANN_assests\scaler_y.pkl"
)

# Create labels
labels = ['Longitude', 'Latitude', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'N', 'Cab', 'Ccx', 'Cw', 'Cm']

X_labels = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

output_labels = ["N", "Cab", "Ccx", "Cw", "Cm"]

layer_names = ["structure parameter", "Chlorophylla+b content (µg/cm2)", "Carotenoids content (µg/cm2)", "Equivalent Water content (cm)", "Leaf Mass per Area (g/cm2)"]

data_to_map = {
    "structure parameter": "N",
    "Chlorophylla+b content (µg/cm2)": "Cab",
    "Carotenoids content (µg/cm2)": "Ccx",
    "Equivalent Water content (cm)": "Cw",
    "Leaf Mass per Area (g/cm2)": "Cm"
}

# gradient_settings = {
#     "structure parameter": {0: 'blue', 0.6: 'cyan', 1.0: 'lime'},
#     "Chlorophylla+b content (µg/cm2)": {0: 'green', 0.6: 'lime', 1.0: 'yellow'},
#     "Carotenoids content (µg/cm2)": {0: 'orange', 0.6: 'red', 1.0: 'maroon'},
#     "Equivalent Water content (cm)": {0: 'navy', 0.6: 'blue', 1.0: 'aqua'},
#     "Leaf Mass per Area (g/cm2)": {0: 'purple', 0.6: 'fuchsia', 1.0: 'pink'}
# }

# gradient_settings = {
#     "structure parameter": {0.2: 'rgba(0, 0, 255, 1.0)', 0.6: 'rgba(0, 255, 255, 1.0)', 1.0: 'rgba(0, 255, 0, 1.0)'},
#     "Chlorophylla+b content (µg/cm2)": {0.2: 'rgba(0, 128, 0, 1.0)', 0.6: 'rgba(127, 255, 0, 1.0)', 1.0: 'rgba(255, 255, 0, 1.0)'},
#     "Carotenoids content (µg/cm2)": {0.2: 'rgba(255, 69, 0, 1.0)', 0.6: 'rgba(255, 0, 0, 1.0)', 1.0: 'rgba(139, 0, 0, 1.0)'},
#     "Equivalent Water content (cm)": {0.2: 'rgba(0, 0, 139, 1.0)', 0.6: 'rgba(65, 105, 225, 1.0)', 1.0: 'rgba(0, 191, 255, 1.0)'},
#     "Leaf Mass per Area (g/cm2)": {0.2: 'rgba(75, 0, 130, 1.0)', 0.6: 'rgba(148, 0, 211, 1.0)', 1.0: 'rgba(255, 20, 147, 1.0)'}
# }

gradient_parula = {
    0.0: 'rgba(128, 0, 128, 1.0)',     # purple
    0.2: 'rgba(0, 0, 255, 1.0)',       # blue
    0.4: 'rgba(0, 255, 255, 1.0)',     # cyan
    0.5: 'rgba(0, 250, 154, 1.0)',     # mediumspringgreen
    0.6: 'rgba(50, 205, 50, 1.0)',     # lime
    0.7: 'rgba(173, 255, 47, 1.0)',    # greenyellow
    0.8: 'rgba(255, 255, 0, 1.0)',     # yellow
    0.9: 'rgba(255, 165, 0, 1.0)',     # orange
    1.0: 'rgba(255, 0, 0, 1.0)'        # red
}

gradient_settings = {
    "structure parameter": gradient_parula,
    "Chlorophylla+b content (µg/cm2)": gradient_parula,
    "Carotenoids content (µg/cm2)": gradient_parula,
    "Equivalent Water content (cm)": gradient_parula,
    "Leaf Mass per Area (g/cm2)": gradient_parula
}


print_with_line_number("Finish loading the ANN model!")

def runModel(input_data, scaler_X, scaler_Y, ANNmodel):
    # Preprocess the Input Data
    # Scale the input features using the previously saved scaler for X
    input_data_scaled = scaler_X.transform(input_data)

    # Use the Model for Prediction
    # Predict the output values (N, Cab, Ccx, Cw, Cm) for each pixel block
    output_data_scaled = ANNmodel.predict(input_data_scaled)

    # Post-process the Output Data
    # Inverse scale the output data using the previously saved scaler for Y
    output_data = scaler_Y.inverse_transform(output_data_scaled)

    # Organize the Output Results and Coordinates
    # Create datasets for each output label and one for coordinates
    # Each dataset contains corresponding data for all pixel blocks
    datasets = {}
    for i, label in enumerate(output_labels):
        datasets[label] = output_data[:, i]

    # Print the results for verification
    for label, data in datasets.items():
        print(label, data)
    
    return datasets

# Use your own Google Map token
def getGPS():
    GPSurl = 'https://www.googleapis.com/geolocation/v1/geolocate?key=Google Map token'
    data = {'homeMobileCountryCode': 310, 'homeMobileNetworkCode': 410, 'considerIp': 'True'}
    response = requests.post(GPSurl, data=json.dumps(data))
    result = json.loads(response.content)
    return result

def get_location(lat, lon):
    geolocator = Nominatim(timeout=120, user_agent="when-to-fly")
    location = geolocator.reverse(f"{lat},{lon}")
    return location.address

app_ui = ui.page_fluid(
    ui.div(
        ui.strong("Tips:"),
        ui.br(),
        ui.span("1.Click the polygon icon on the map to draw a polygon, the circular icon to mark a location, the line icon to measure distance, and the icon in the top left corner of the map to select the layers you want to display."),
        ui.br(),
        ui.span("2.After selecting an area, click the 'Analyze' button to analyze the leaf-level feature data for that area. The results are presented as heat maps, with brighter areas indicating values closer to the maximum."),
        ui.br(),
        ui.span("3.Currently, the analysis does not support multiple polygons. The application will only recognize the last polygoned area."),
        ui.br(),
        ui.span("4.After analyzing the data of the drawn area, the webpage may experience slower loading speeds and delays. Please be patient and wait after performing an operation."),
        ui.br(),
        ui.strong("If you are unable to zoom in or out of the map using the mouse scroll wheel, please use the slide bar provided above to zoom directly.", style="color: green;"),
        ui.br(),
        ui.strong("We strongly recommend that you use a smaller scale to view the heat map (17 or 18 zoom level), as it will retain more details.", style="color: red;")
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.div(
                ui.div(
                    ui.input_date("date", "Date:"), 
                    ui.input_slider("zoom", "Map zoom level", value=12, min=1, max=18),
                ),
                ui.div(
                    ui.input_numeric("lat", "Latitude", value=38.53667742),
                    ui.input_numeric("long", "Longitude", value=-121.75387309),
                ),
                style=css(display="flex", justify_content="center", align_items="center", gap="1rem"),
            ),
        ),
        ui.panel_main(
            ui.div( 
                ui.output_text("N_range"),
                ui.output_text("Cab_range"),
                ui.output_text("Ccx_range"),
                ui.output_text("Cw_range"),
                ui.output_text("Cm_range"),
                style=css(display="flex", justify_content="center", align_items="center", gap="2rem"),
            ),
            ui.img(src="legend.png"),
        ),
    ),
    output_widget("map"),
    ui.strong("Must analyze (to renew the image information) before downloading any file."),
    ui.div(
        ui.input_action_button("analyze", "Analyze", class_="btn-success"),
        ui.download_button("download_polygon", "Download spectral data as tif", class_="btn-success"),
        ui.download_button("download_output", "Download spectral and output data as csv", class_="btn-success"),
        style=css(display="flex", justify_content="center", align_items="center", gap="2rem"),
    ),
)

# re-run when a user using the application
def server(input, output, session):
    m = ui.modal(
        "Please wait for progress...",
        easy_close=False,
        size="s",
        footer=None,
        fade=True
    )
    ui.modal_show(m)
    # Initialize Earth Engine
    # ee.Initialize(credentials)
    ee.Initialize()
    # Check API status
    asset_roots = ee.data.getAssetRoots()
    if asset_roots:
        print("Active Project ID:", asset_roots[0]['id'])
        print("API is connected and working: ", asset_roots)
    else:
        print("API is not connected or not working.")

    global address_line, polygoned_image, center_flag, loc_flag, output_df
    address_line = None
    polygoned_image = None
    center_flag = False
    loc_flag = False
    output_df = pd.DataFrame()
    polygon_data = reactive.Value([])
    N = reactive.Value("structure parameter")
    Cab = reactive.Value("Chlorophylla+b content (µg/cm2)")
    Ccx = reactive.Value("Carotenoids content (µg/cm2)")
    Cw = reactive.Value("Equivalent Water content (cm)")
    Cm = reactive.Value("Leaf Mass per Area (g/cm2)")

    @output
    @render.text
    def N_range():
        return N.get()

    @output
    @render.text
    def Cab_range():
        return Cab.get()
    
    @output
    @render.text
    def Ccx_range():
        return Ccx.get()
    
    @output
    @render.text
    def Cw_range():
        return Cw.get()
    
    @output
    @render.text
    def Cm_range():
        return Cm.get()

    def handle_draw(self, action, geo_json):
        print("运行handle_draw")
        if geo_json['type'] == 'Feature':
            # Check if the drawn shape is a polygon
            if geo_json['geometry']['type'] == 'Polygon':
                # Get the coordinates of the polygon's vertices
                coordinates = geo_json['geometry']['coordinates'][0]

                # Extract latitude and longitude values from each vertex
                # For GeoJSON, coordinates are represented as [longitude, latitude]
                # (note the reverse order compared to traditional [latitude, longitude])
                polygon_data.set([(lon, lat) for lon, lat in coordinates])

                # Process the polygon_data as per your requirement
                # For example, print the coordinates
                print("Polygon Vertex Coordinates:")
                for lon, lat in polygon_data.get():
                    print(f"Latitude: {lat}, Longitude: {lon}")

    try:
        # Get the user's current geoinformation
        current_gps = getGPS()
        print_with_line_number(current_gps)
        current_location = get_location(current_gps['location']['lat'],  current_gps['location']['lng'])
        print_with_line_number(current_location)
        ui.update_text(id="address",
                       label="Data for",
                       value=current_location)
        
        # Initialize and display when the session starts (1)
        map = L.Map(center=(current_gps['location']['lat'],  current_gps['location']['lng']), zoom=12, scroll_wheel_zoom=True)
        map.layout = Layout(height='600px')

        @reactive.isolate()
        def update_text_inputs(lat: Optional[float], long: Optional[float]) -> None:
            req(lat is not None, long is not None)
            lat = round(lat, 8)
            long = round(long, 8)
            if lat != input.lat():
                input.lat.freeze()
                ui.update_text("lat", value=lat)
            if long != input.long():
                input.long.freeze()
                ui.update_text("long", value=long)

        # center_flag = True
        # update_text_inputs(current_gps['location']['lat'], current_gps['location']['lng'])
        # center_flag = False

        map.add_layer(L.TileLayer(url='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', name='Natural Map'))
        
        # Add a distance scale
        map.add_control(L.leaflet.ScaleControl(position="bottomleft"))
        layer_control = L.LayersControl(position='topright')
        map.add_control(layer_control)

        # Add the DrawControl widget to the map
        draw_control = L.DrawControl(
            polygon = {
                "shapeOptions": {
                    "fillColor": "transparent",  
                    "fillOpacity": 0.0  
                }
            }
        )
        map.add_control(draw_control)
        # Attach the handle_draw function to the on_draw event
        draw_control.on_draw(handle_draw)
        register_widget("map", map)

        ui.modal_remove()

    except Exception as e:
        ui.modal_remove()
        error_modal = ui.modal(
            str(e),
            title="An Error occured. Please refresh",
            easy_close=True,
            size="xl",
            footer=None,
            fade=True
        )
        # print_with_line_number("Show error modal")
        ui.modal_show(error_modal)
        traceback.print_exc()

    # When the slider changes, update the map's zoom attribute (2)
    @reactive.Effect
    def _():
        map.zoom = input.zoom()

    # When zooming directly on the map, update the slider's value (2 and 3)
    @reactive.Effect
    def _():
        ui.update_slider("zoom", value=reactive_read(map, "zoom"))

    @reactive.Effect
    def location():
        """Returns tuple of (lat,long) floats--or throws silent error if no lat/long is
        selected"""
        # Require lat/long to be populated before we can proceed
        req(input.lat() is not None, input.long() is not None)

        try:
            long = input.long()
            # Wrap longitudes so they're within [-180, 180]
            long = (long + 180) % 360 - 180
            if round(map.center[0], 8) == input.lat() and round(map.center[1], 8) == long:
                return
            map.center = (input.lat(), long)

        except ValueError as e:
            error_modal = ui.modal(
                str(e),
                title="Invalid latitude/longitude specification. Please refresh",
                easy_close=True,
                size="xl",
                footer=None,
                fade=True
            )
            # print_with_line_number("Show error modal")
            ui.modal_show(error_modal)
            traceback.print_exc()

    # Everytime the map's bounds change, update the output message (3)
    # rerun when a user do some reactive changes.
    # @output
    # @render.ui
    @reactive.Effect
    def map_bounds():
        center = reactive_read(map, "center")
        if len(center) == 0:
            return        
        lon = (center[1] + 180) % 360 - 180
        update_text_inputs(center[0], lon)
    
    def update_or_create_heatmaps(output_datasets, scale):
        """
        Check if a heatmap layer exists for each dataset in output_datasets.
        If it exists, update the heatmap, otherwise create a new heatmap.
        
        Parameters:
            output_datasets (list of dict): The datasets for creating/updating heatmaps
        """
        # Iterate over each dataset in output_datasets
        existing_layers = {layer.name: layer for layer in map.layers}

        for layer_name in layer_names:
            # Check if a heatmap layer with this name already exists
            if layer_name in existing_layers:
                print("deleting ", layer_name)
                map.remove_layer(existing_layers[layer_name])

            heatmap_data = []
            data_values = output_datasets[data_to_map[layer_name]]

            q03 = np.percentile(data_values, 3)
            q97 = np.percentile(data_values, 97) 
            min_value = min(data_values)
            if min_value < 0:
                min_value = 0
            max_value = max(data_values)

            if (data_to_map[layer_name] == "N"):
                N.set(layer_name + ": " + str(min_value) + " ~ " + str(max_value))
            elif (data_to_map[layer_name] == "Cab"): 
                Cab.set(layer_name + ": " + str(min_value) + " ~ " + str(max_value))
            elif (data_to_map[layer_name] == "Ccx"): 
                Ccx.set(layer_name + ": " + str(min_value) + " ~ " + str(max_value))
            elif (data_to_map[layer_name] == "Cw"): 
                Cw.set(layer_name + ": " + str(min_value) + " ~ " + str(max_value))
            else: 
                Cm.set(layer_name + ": " + str(min_value) + " ~ " + str(max_value))
                
            for coord, n in zip(output_datasets["Coordinates"], data_values):
                # normalized_value = (n - min_value) / (max_value - min_value)
                if n <= q03:
                    normalized_value = 0
                elif n >= q97:
                    normalized_value = 1
                else:
                    normalized_value = (n - q03) / (q97 - q03)
                heatmap_data.append([coord[1], coord[0], normalized_value])
                     
            # Generate new heatmap for this dataset
            heatmap = L.Heatmap(
                locations=heatmap_data,
                radius=scale * 1.3,
                # gradient=gradient_settings[layer_name],
                gradient=gradient_parula,
                max=1,
                blur=scale / 2,
                name=layer_name
            )

            # Add the new heatmap layer to the map
            map.add_layer(heatmap)
    
    @reactive.Effect
    @reactive.event(input.analyze, ignore_none=True, ignore_init=True)
    def _():
        global output_df
        if not polygon_data.get():
            return
        ui.modal_show(m)
        global polygoned_image
        polygon = ee.Geometry.Polygon(polygon_data.get())
        print("Polygon Data: " , polygon_data.get())
        print("Polygon: " , polygon)
        
        # Define Sentinel-2 image collection ("2021-01-01", "2021-12-31")
        current_date = input.date()
        today = ee.Date(input.date().strftime('%Y-%m-%d')) 
        start_date = today.advance(-15, 'day')

        print("Start Date: ", start_date.format('YYYY-MM-dd').getInfo(), "| End Date: ", today.format('YYYY-MM-dd').getInfo())

        sentinel2 = ee.ImageCollection("COPERNICUS/S2_SR")\
                    .filterDate(start_date, today)\
                    .filterBounds(polygon)\
                    .sort('CLOUDY_PIXEL_PERCENTAGE', True)
                    # .first()  # Retrieve the first image from the ImageCollection
        # sentinel2 = sentinel2.sort('CLOUDY_PIXEL_PERCENTAGE', True)
        polygoned_image = sentinel2.first()

        # Make sure the polygoned_image is available
        if not polygoned_image:
            print("No image available for download.")
            return

        retry = 5
        while(sentinel2.size().getInfo() == 0):
            if(retry == 0):
                ui.update_date("date", label="Date:", value=current_date)
                ui.modal_remove()
                error_modal = ui.modal(
                    "Please do not choose a date that is too far in the future. This application will search for remote sensing data within the two weeks prior to the selected date.",
                    title="Something wrong happened, try \"Analyze\" again ",
                    easy_close=True,
                    size="xl",
                    footer=None,
                    fade=True
                )
                ui.modal_show(error_modal)
                print("fail to fecth image.")
                return
            print("wait for fetching.")
            time.sleep(2)
            retry -= 1


        print_with_line_number("Type of sentinel2: " + str(type(sentinel2)))
        print("Counts of Fetched image: ", sentinel2.size().getInfo())

        # Clip the image to the extent of the polygon
        clipped_image = polygoned_image.clip(polygon)

        # Get meta data about the image object
        # bands = clipped_image.bandNames().getInfo()
        # print_with_line_number(bands)

        # Calculate suitable pixel number. GEE service allow fecthing 5000 pixels at most for one call. So we use the "polygoned area / 4999" to decide a rational pixel scale.
        scale=1
        polygon_area = polygon.area().getInfo()
        num = math.ceil(polygon_area / scale / scale)
        if (num > 4999):
            per_area = math.ceil(polygon_area / 4998)
            scale = math.ceil(math.pow(per_area, 1.0/2))

        print("polygon_area(m2): ", polygon_area, "scale: ", scale)

        # Fetch reflectance of B1-B12
        spectral_values = clipped_image.select('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12').sample(
            region=polygon,
            scale=scale,
            numPixels=4999, 
            geometries=True
        )

        print_with_line_number("Pre-process the bands data.")
        # print_with_line_number(type(spectral_values))
        spectral_values = spectral_values.getInfo()
        # print_with_line_number(type(spectral_values))
        spectral_values_json = json.dumps(spectral_values)
        # print(spectral_values_json)
        spectral_values_dict = json.loads(spectral_values_json)
        features = spectral_values_dict['features']

        print_with_line_number("Extract the center coordinates and values of B1-B12 for each pixel block")
        coords = []
        input_data = []
        for feature in features:
            coords.append(feature['geometry']['coordinates'])
            props = feature['properties']
            input_data.append([props[b] for b in X_labels])

        # Convert to NumPy arrays
        coords = np.array(coords)
        print("coords: ", coords)
        input_data = np.array(input_data)
        print("input_bands: ", input_data)

        output_datasets = runModel(input_data, loaded_scaler_X, loaded_scaler_Y, model)

        print_with_line_number("Add a dataset for the coordinates")
        # Combine all data
        data_combined = np.column_stack((coords, input_data, output_datasets['N'], output_datasets['Cab'], output_datasets['Ccx'], output_datasets['Cw'], output_datasets['Cm']))
        # Convert to DataFrame
        output_df = pd.DataFrame(data_combined, columns=labels)
        
        output_datasets['Coordinates'] = coords
        update_or_create_heatmaps(output_datasets, scale)
        register_widget("map", map)
        ui.modal_remove()
    
    @reactive.Effect
    def _():
        print("Current navbar page: ", input.navbar_id())
    
    @session.download(
        filename=lambda: f"image-{input.date().isoformat()}-{np.random.randint(100, 999)}.tif"
    )
    def download_polygon():
        # # Replace this with your ee.Image object
        # image_id = "COPERNICUS/S2_SR/20230728T184921_20230728T190044_T10SFH"
        # image = ee.Image(image_id)
        # Make sure the polygoned_image is available
        if not polygoned_image:
            print("No image available for download.")
            return

        # Clip the image to the extent of the polygon
        clipped_image = polygoned_image.clip(ee.Geometry.Polygon(polygon_data.get()))
        print("clipped_image: ", clipped_image)

        # Define export parameters
        download_params = {
            'scale': 10,
            'region': polygon_data.get(), # ee.Geometry object defining the region to export
            'format': 'GeoTIFF',
        }

        # Generate download URL for the GeoTIFF image
        download_url = clipped_image.getDownloadURL(download_params)

        # Send a request to download the image
        response = requests.get(download_url)

        # Create a BytesIO buffer
        with io.BytesIO() as buf:
            # Write the image content to the buffer
            buf.write(response.content)
            buf.seek(0)  # Move the buffer's position to the beginning

            # Yield the buffer's content as a downloadable file
            yield buf.getvalue()

        print("Image downloaded successfully!")

    @session.download(
        filename=lambda: f"data-{input.date().isoformat()}-{np.random.randint(100, 999)}.csv"
    )
    def download_output():
        global output_df
        # Check if data is available
        if output_df.empty:
            print("No data available for download.")
            return
        
        # Convert dataframe to CSV and encode to bytes
        csv_data = output_df.to_csv(index=False).encode()
        
        # Create a StringIO buffer for textual data
        with io.BytesIO() as buf:
            buf.write(csv_data)
            # Reset the buffer's position to the beginning
            buf.seek(0)
            # Create and return a streaming response
            yield buf.getvalue()

        print("Data downloaded successfully!")
    
static_dir = Path(__file__).parent / "assets"
app = App(app_ui, server, static_assets=static_dir)
