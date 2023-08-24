import os
import sys
from typing import Optional
from shiny import App, render, ui, reactive, req
import ipyleaflet as L
from htmltools import css
import pandas as pd
import numpy as np
from shinywidgets import output_widget, reactive_read, register_widget
from geopy.geocoders import Nominatim
import json
import requests
import traceback
import io
import asyncio
import plotly.graph_objects as go
from datetime import datetime, date
import pytz
from typing import List
from shiny.types import NavSetArg
from utils import print_with_line_number, datafields
from timezonefinder import TimezoneFinder

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

tf = TimezoneFinder()
CHOICES = ["cloudBase", "cloudCeiling", "dewPoint", "evapotranspiration", "freezingRainIntensity", "humidity", "iceAccumulation", "pressureSurfaceLevel", "rainAccumulation", "rainIntensity", "sleetAccumulation", "sleetIntensity", "snowAccumulation", "snowDepth", "snowIntensity", "temperature", "temperatureApparent", "uvHealthConcern", "uvIndex", "visibility", "windDirection", "windGust", "windSpeed"]

def getGPS():
    GPSurl = 'https://www.googleapis.com/geolocation/v1/geolocate?key=AIzaSyAnHc2yRD53vlzHrj7qQ6OLFiX-iGsqFyM'
    data = {'homeMobileCountryCode': 310, 'homeMobileNetworkCode': 410, 'considerIp': 'True'}
    response = requests.post(GPSurl, data=json.dumps(data))
    result = json.loads(response.content)
    return result

def handle_draw(action, geo_json, data_holder):
    if geo_json['type'] == 'Feature':
        # Check if the drawn shape is a polygon
        if geo_json['geometry']['type'] == 'Polygon':
            # Get the coordinates of the polygon's vertices
            coordinates = geo_json['geometry']['coordinates'][0]

            # Extract latitude and longitude values from each vertex
            # For GeoJSON, coordinates are represented as [longitude, latitude]
            # (note the reverse order compared to traditional [latitude, longitude])
            data_holder = [(lat, lon) for lon, lat in coordinates]

            print(data_holder)

# # Function to handle the polygon data received from the frontend
# def handle_polygon(data):
#     # Process and display the data as needed
#     return data

def get_location(lat, lon):
    geolocator = Nominatim(user_agent="when-to-fly")
    location = geolocator.reverse(f"{lat},{lon}")
    return location.address


def nav_controls(prefix: str) -> List[NavSetArg]:
    return [
        # ui.nav("When to Fly",
        # ),
        ui.nav("Custome Garphics",
                ui.panel_title("Customize your weather data"),
                ui.div(
                        ui.input_slider("zoom", "Map zoom level", value=12, min=1, max=18),
                        ui.input_numeric("lat", "Latitude", value=38.53667742),
                        ui.input_numeric("long", "Longitude", value=-121.75387309),
                        ui.help_text("Click to select location"),
                        ui.output_ui("map_bounds"),
                        style=css(
                        display="flex", justify_content="center", align_items="center", gap="2rem"
                    ),
                ),
                output_widget("map"),
                ui.layout_sidebar(
                    ui.panel_sidebar(
                        ui.input_selectize("items", "Select up to 4 items you want to show in your graph", choices=CHOICES, multiple = True),
                        ui.input_action_button("do_plot", "Plot", class_="btn-success"), 
                    ),
                    ui.panel_main(
                        ui.div(
                            ui.strong("cloudCover(%)"),
                            ui.span(": The fraction of the sky obscured by clouds when observed from a particular location. The part with a red shadow in the figure indicates that the value is greater than or equal to 25%, and it is "),
                            ui.strong("not recommended"),
                            " to travel at this time."
                        ),
                        ui.div(
                            ui.strong("precipitationProbability(%)"),
                            ui.span(": Probability of precipitation represents the chance of >0.0254 cm (0.01 in.) of liquid equivalent precipitation at a radius surrounding a point location over a specific period of time.")
                        ),
                        ui.div(
                            ui.a("See all data field details.", href="https://docs.google.com/document/d/1fmiUYToF2YElzNvPT3_Zo8dBc9kzGgWJZGj14yWx2Bo/edit?usp=sharing")
                        ),
                        ui.output_ui("info_html")
                    )
               ),
               output_widget("plot_weather"),       
               ),
        ui.nav("Download data",
               ui.input_text(id="address", label="Data for", value="", width='100%'),
               ui.output_data_frame("weather_frame"), 
               ui.download_button("download_weather", "Download Data as csv", class_="btn-success"),
               ),
        ui.nav("Legal Area",
        ),
        ui.nav_spacer(),
        ui.nav_menu(
            "Other links",
            ui.nav_control(
                ui.a(
                    "shiny for Python",
                    href="https://rstudio.com",
                    target="_blank",
                )
            ),
            ui.nav_control(
                ui.a(
                    "tomorrow.io(weather data)",
                    href="https://rstudio.com",
                    target="_blank",
                )
            ),
            align="right",
        ),
    ]

app_ui = ui.page_fluid(
    ui.page_navbar(
        *nav_controls("page_navbar"),
        title="My Views",
        bg="#006400",
        inverse=True,
        id="navbar_id",
        footer=ui.div(
            {"style": "width:80%;margin: 0 auto"},
            ui.tags.style(
                """
                h4 {
                    margin-top: 3em;
                }
                """
            ),
            # ui.navset_pill_card(*nav_controls("navset_pill_card()")),
        )
    )
)

# re-run when a user using the application
def server(input, output, session):
    global weather_data, remap_flag, address_line, weather_fig, m
    weather_data = None
    remap_flag = False
    address_line = None
    weatherframe = reactive.Value(pd.DataFrame())

    def check(weather_data, lat, lon):
        # Get current local Time
        timezone = tf.timezone_at(lng=lon, lat=lat)
        current_time = datetime.now(pytz.timezone(timezone))

        def convert_to_local_time(utc_time):
            dt = datetime.strptime(utc_time, "%Y-%m-%dT%H:%M:%SZ")
            local_time = dt.astimezone(pytz.timezone(timezone))
            return local_time.strftime("%Y-%m-%d %H")

        # # Check if it's on time locally
        # if(weather_data == None or (current_time.minute == 0 and current_time.second == 0)):
        weather_url = f"https://api.tomorrow.io/v4/weather/forecast?location={lat},{lon}&apikey=Purg6j6hjn9LdzMVwRvToPbJVhnlSjAP"
        response = requests.get(weather_url)
        if response.status_code != 200:
            raise Exception(f"Error fetching {weather_url}: {response.status_code}")
        api_data = response.json()
        hourly_data = api_data['timelines']['hourly']
        weather_data = pd.DataFrame([{**{'time': item['time']}, **item['values']} for item in hourly_data])
        weather_data.fillna(0, inplace=True)

        # Using convert_to_local_time to process the column 'time' data into local datetime string
        weather_data['time'] = weather_data['time'].apply(convert_to_local_time)
        weather_data['datetime'] = weather_data['time'].copy()
        weather_data.set_index('datetime', inplace=True)

        print_with_line_number("Weather dataframe:")
        # print(weather_data.shape)
        # print(weather_data.head(5))
        weatherframe.set(weather_data)
        
        return weather_data


    m = ui.modal(
            "Please wait for progress...",
            easy_close=False,
            size="s",
            footer=None,
            fade=True
        )


    def plot_weather(fig, weather_data, items) -> go.Figure:
        # if map_initialized:
        #     print_with_line_number("Show plotting modal")
        #     ui.modal_show(m)             
        time = weather_data['time'].values

        fig.update_layout(
            xaxis_title='Datetime',
        )

        cloud_coverage = weather_data["cloudCover"].copy()
        cloud_coverage[cloud_coverage < 0.25] = 0

        fig.add_trace(go.Scatter(
            x = time,
            y = cloud_coverage,
            name = "cloudCover",
            fill='tozeroy',
            marker_color ='indianred', 
            opacity = 0.3
            ))
        
        fig.update_layout(**{"yaxis": {"title":"cloudCover(%)", "side":"left"}}, overwrite=False)

        
        fig.add_trace(go.Bar(
            x = time,
            y = weather_data["precipitationProbability"].values,
            name = "precipitationProbability",
            yaxis = "y2",
            marker_color ='rgb(158,202,225)', 
            marker_line_color = 'rgb(8,48,107)',
            marker_line_width = 1.5, 
            opacity = 0.6
        ))

        fig.update_layout(**{"yaxis2": 
                                {
                                "title":"precipitationProbability(%)",
                                "anchor": "free",
                                "overlaying": "y",
                                "side": "right",
                                "autoshift": True,
                                    }}
                                    , overwrite=False)

        count = 3
        pos = ["right", "left"]
        for item in items:

            y_axis_key = f"yaxis{count}"
            yname = f"y{count}"

            fig.add_trace(go.Scatter(
                x = time,
                y = weather_data[item].values,
                name = item,
                yaxis = yname
            ))

            y_axis_params = dict(
                title = datafields[item],
                anchor="free",
                overlaying="y",  # 将overlaying属性设置为None，避免y轴之间重叠
                side = pos[count % 2],
                autoshift=True,
            )
            
            fig.update_layout(**{y_axis_key: y_axis_params}, overwrite=False)
            count += 1

        # Adjust legend position to the top # Set a default height
        fig.update_layout(legend=dict(y=1.1, yanchor="top", orientation="h"), height=800)

        # print_with_line_number(fig)
        
        # if map_initialized:
        #     print_with_line_number("Remove plotting modal")
        #     ui.modal_remove()

        return fig
    
        
    try:
        print_with_line_number("Show initializing modal")
        ui.modal_show(m)
        map_initialized = False
        
        # Initialize and display when the session starts (1)
        map = L.Map(center=(38.53667742,  -121.75387309), zoom=12, scroll_wheel_zoom=True)
        map.add_layer(L.TileLayer(url='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', name='Natural Map'))
        with reactive.isolate():
            marker = L.Marker(location=(input.lat() or 38.53667742 , input.long() or -121.75387309), name='Marker')
        control = L.LayersControl(position='topright')
        map.add_control(control)

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
            map.center = (lat, long)

        @reactive.isolate()
        def update_marker(lat: Optional[float], long: Optional[float]) -> None:
            req(lat is not None, long is not None)
            lat = round(lat, 8)
            long = round(long, 8)
            if marker.location != (lat, long):
                marker.location = (lat, long)
            if marker not in map.layers:
                map.add_layer(marker)
            map.center = marker.location
        
        @reactive.Effect
        def sync_inputs_to_marker():
            update_marker(input.lat(), input.long())

        def on_map_interaction(**kwargs):
            if kwargs.get("type") == "click":
                lat, long = kwargs.get("coordinates")
                update_text_inputs(lat, long)


        # Get the user's current geoinformation
        current_gps = getGPS()
        update_text_inputs(current_gps['location']['lat'], current_gps['location']['lng'])

        # ui.update_numeric("lat", value=current_gps['location']['lat'])
        # ui.update_numeric("long", value=current_gps['location']['lng'])
        # print("Input: ", input.lat(),  input.long())

        print_with_line_number(current_gps)
        current_location = get_location(current_gps['location']['lat'],  current_gps['location']['lng'])
        print_with_line_number(current_location)
        ui.update_text(id="address",
                       label="Data for",
                       value=current_location)


        map.on_interaction(on_map_interaction)
        # Add a distance scale
        map.add_control(L.leaflet.ScaleControl(position="bottomleft"))
        register_widget("map", map)


        #  Fetch weather data
        # await check(weather_data, current_gps['location']['lat'],  current_gps['location']['lng'])
        print_with_line_number(weather_data)
        weather_data = check(weather_data, current_gps['location']['lat'],  current_gps['location']['lng'])
        # choices = weather_data.columns.tolist()
        # print(choices)
        print_with_line_number("Finish fetching hourly data!")

        # In your server function, create the initial fig
        weather_fig = go.Figure()

        # Call plot_weather to initialize the plot
        plot_weather(weather_fig, weather_data, [])
        register_widget("plot_weather", weather_fig)

        print_with_line_number("Finish plotting selected data!")

        map_initialized = True

        print_with_line_number("Remove initializing modal")
        ui.modal_remove()

    except Exception as e:
        ui.modal_remove()
        error_modal = ui.modal(
            str(e),
            title="An Error occured, Please refresh",
            easy_close=True,
            size="xl",
            footer=None,
            fade=True
        )
        # print_with_line_number("Show error modal")
        ui.modal_show(error_modal)
        traceback.print_exc()

    @reactive.Calc
    def location():
        """Returns tuple of (lat,long) floats--or throws silent error if no lat/long is
        selected"""

        # Require lat/long to be populated before we can proceed
        req(input.lat() is not None, input.long() is not None)

        try:
            long = input.long()
            # Wrap longitudes so they're within [-180, 180]
            long = (long + 180) % 360 - 180
            return (input.lat(), long)
        except ValueError:
            raise ValueError("Invalid latitude/longitude specification")

    # When the slider changes, update the map's zoom attribute (2)
    @reactive.Effect
    def _():
        if not map_initialized:
            return
        map.zoom = input.zoom()

    # When zooming directly on the map, update the slider's value (2 and 3)
    @reactive.Effect
    def _():
        if not map_initialized:
            return
        ui.update_slider("zoom", value=reactive_read(map, "zoom"))

    
    @reactive.Effect
    def _():
        print("Current navbar page: ", input.navbar_id())

        # Everytime the map's bounds change, update the output message (3)
    # rerun when a user do some reactive changes.
    @reactive.Effect
    async def map_bounds():
        if not map_initialized:
            return
        global weather_data, remap_flag
        print("Change bounds")
        center = location()

        # center = reactive_read(map, "center")
        # if len(center) == 0:
        #     return     
        # lat = round(center[0], 4)
        # lon = (center[1] + 180) % 360 - 180
        # lon = round(lon, 4)

        # print_with_line_number("Some weather data")
         # print(center, lat, lon)
        if (remap_flag):
            # print_with_line_number("remap weather_data")
            weather_data = check(weather_data, center[0],  center[1])
            # print_with_line_number("Updating hourly data!")
            update_plot()
        remap_flag = True

        new_location = get_location(center[0],  center[1])
        ui.update_text(id="address",
                       label="Data for",
                       value=new_location)

        # return ui.p(f"Latitude: {lat}", ui.br(), f"Longitude: {lon}")
    
    def update_plot():
        global weather_fig, weather_data
        # Assuming you have updated the weather_data with new data
        # For example: weather_data = updated_weather_data()
        # Call plot_weather to update the plot with the new weather_data
        weather_fig = go.Figure()
        plot_weather(weather_fig, weather_data, input.items())
        register_widget("plot_weather", weather_fig)


    @reactive.Effect
    @reactive.event(input.items)
    def _():
        global remap_flag
        remap_flag = False
        transfer = list(input.items())
        if (len(transfer) > 4 ):
            transfer.pop()
            ui.notification_show("At most four options can be selected!", type="warning")
            ui.update_selectize(
            "items",
            choices = CHOICES,
            selected=transfer,
            server=True,
            )
        # print(input.items())

    @output
    @render.data_frame
    async def weather_frame():
        return weatherframe.get()
    
    @session.download(
        filename=lambda: f"data-{date.today().isoformat()}-{np.random.randint(100,999)}.csv"
    )
    async def download_weather():
        # This version uses a function to generate the filename. It also yields data
        # multiple times.
        await asyncio.sleep(0.25)
            # Create a BytesIO buffer
        with io.BytesIO() as buf:
            # Write the DataFrame to the buffer as CSV
            weather_data.to_csv(buf, index=False)
            buf.seek(0) # Move the buffer's position to the beginning

            # Return the buffer's content as a downloadable file
            yield buf.getvalue()
    
    # Use reactive.event() to invalidate the plot only when the button is pressed
    # (not when the slider is changed)
    @reactive.Effect
    @reactive.event(input.do_plot, ignore_none=True, ignore_init=True)
    def _():
        global remap_flag
        # print_with_line_number("In revisving")
        update_plot()
        remap_flag = True

    
app = App(app_ui, server)