import numpy as np
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui
# import cv2
from datetime import datetime,  timedelta, date, time
from pvlib import solarposition
import pvlib
import pytz
from typing import Dict, List, Optional, Tuple
# from PIL import Image
import astropy.units as u
# import matplotlib.dates as mpldates
# from solartime import SolarTime
import pandas as pd
# import ephem
import suntime
import timezonefinder
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
# Import some library related to ploting
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors



from low_cation import location_server, location_ui

# pd.set_option('display.max_columns', None)
pd.options.display.max_rows = 100
pd.options.display.max_columns = 80

class CustomError(Exception):
    def __init__(self, message):
        self.message = message

cam_types = ['MicaSense RedRdge','MicaSense Altum-PT','DJI P4M','Parrot Sequoia','Sentera 6X','YuSense MS600 Pro','Other']
cam_HVD_FOV={'MicaSense Altum-PT': [48, 36.8,60.5],'MicaSense RedRdge':[47.2, 35.4,59]
           ,'DJI P4M':[50.6 , 42,65.7],'Parrot Sequoia':[70.6,50.6,86.8],'Sentera 6X':[47,35.2,58.7],
           'YuSense MS600 Pro':[43.6,33.4,54.9]}

# import math
# for key in cam_HVD_FOV.keys():
#     math.sqrt(cam_HVD_FOV[key][0]**2+cam_HVD_FOV[key][1]**2)
#     print(cam_HVD_FOV[key][0],cam_HVD_FOV[key][1],math.sqrt(cam_HVD_FOV[key][0]**2+cam_HVD_FOV[key][1]**2))

cam_XYratio={'MicaSense Altum-PT': [2064 , 1544],'MicaSense RedRdge':[1280 , 960],
             'DJI P4M':[1600,1300 ],'Parrot Sequoia':[1280,960],'Sentera 6X':[2048,1536],
             'YuSense MS600 Pro':[1280,960]}

app_ui = ui.page_fixed(
    ui.tags.h3("Hotspot & Weather calculator - A tool for UAV flight time planning"),
    ui.div(
        ui.markdown(
            """This appliction uses [the information published in the ISPRS journal](https://www.sciencedirect.com/science/article/pii/S0924271622003161) to calculate the
            hotspot and darkspt occurance in areal images based the location of the field,
            FOV of the camera, and flight date. 
            The green area in chart below shows the best time frame to fly the UAV
            with the selected camera to avoid having hot or dark spot inside image.
            """
        ),
        class_="mb-5",
    ),
    ui.row(
        ui.column(
            8,
            ui.output_ui("timeinfo"),
            ui.output_plot("plot", height="500px"),
            ui.output_ui("Print_result"),
            ui.output_plot("weather_plot", height="500px"),
            ui.output_ui("weather_result"),
            # For debugging
            # ui.output_table("table"),
            # ui.output_image('im_plots'),
            # ui.output_text_verbatim("Print_result"),
            # ui.markdown("""This applict"""),
 
            class_="order-2 order-sm-1",
        ),
        ui.column(
            4,
            ui.panel_well(
                ui.input_date("date", "Flight Date"),
                class_="pb-1 mb-3",
            ),
            ui.input_selectize("FOV", "Camera Type", cam_types),
            ui.output_text_verbatim("Print_FOV"),
            # ui.output_text_verbatim("Print_df"),
            # ui.panel_well(
            #     ui.input_text_area(
            #         "objects", "Camera Info", "M1, NGC35, PLX299", rows=1
            #     ),
            #     class_="pb-1 mb-3",
            # ),
            ui.panel_well(
                location_ui("location"),
                class_="mb-3",
            ),
            class_="order-1 order-sm-2",
        ),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    print("Initializing")
    loc = location_server("location")
    print("loc: ", loc)
    time_padding = timedelta(hours=1.5)
    
    @output
    @render.text
    def Print_FOV():
        cam=input.FOV()
        sp_df=get_day_sp_df()
        # res = next(x for x, val in enumerate(sp_df['hsd']) if val <0)
        if cam in cam_types:
            stri="Camera information: \n Camera FOV (HxVxD):"+ str(cam_HVD_FOV[cam]) +"\n Pixel Ratio:" + str(cam_XYratio[cam])
            return stri
        else:
            return 'res'
        
    @output
    @render.ui
    def weather_result():
        return ui.div(
            ui.div(                '''
                This illustration presents weather factors closely associated with drone operations 
                (spanning from the current day to four days ahead). 
                Time periods with cloud coverage* greater than 25% are highlighted in yellow, 
                segments with potential rainfall* are depicted in blue
                , and instances where both conditions overlap are shown in light green. 
                Droning is not recommended under any of these circumstances.
                Additionally, please assess wind speed and temperature based on your drone's specifications 
                and your personal circumstances to determine if conducting drone operations is feasible\n.
                '''),
            ui.div(
                ui.strong("* Cloud Coverage(%)"),
                ui.span(": The fraction of the sky obscured by clouds when observed from a particular location. The part with a red shadow in the figure indicates that the value is greater than or equal to 25%, and it is "),
                ui.strong("not recommended"),
                    " to travel at this time.",
                ),
            ui.div(
                ui.strong("* Precipitation Probability(%)"),
                ui.span(": Probability of precipitation represents the chance of >0.0254 cm (0.01 in.) of liquid equivalent precipitation at a radius surrounding a point location over a specific period of time.")
            ),
           )

    @output
    @render.text
    def Print_result():
        spdf=get_day_sp_df()
        hsh_zone=[x for x, val in enumerate(spdf['hsh']) if val <0]
        ds=spdf.loc[spdf['dsh']>0]
        amds=ds.head(1)['loc_time'].values[0]
        pmds=ds.tail(1)['loc_time'].values[0]
        if len(hsh_zone)>0:
            hsh_st=spdf['loc_time'][hsh_zone[0]]
            hsh_end=spdf['loc_time'][hsh_zone[-1]]
            recom=f"Based on the selected location, camera type, and date you selected, there could be hotspot and darkspot inside image fram. To avoid hotspots and darkspots, we recommend flying between {amds}-{hsh_st}  OR {hsh_end}-{pmds}"           
        else:
            recom=f"Based on the location, camera type, and date you selected, there will not be any hotspot inside image frame. To avoid darkspots, we recommend flying between {amds} and {pmds}"
        return recom
        
        
    @output
    @render.text
    def Print_df():
        spdf=get_day_sp_df()
        ds=spdf.loc[spdf['dsh']>0]
        amds=float(ds.head(1)['hour'])
        pmds=float(ds.tail(1)['hour']) 
        sr=spdf.head(1)['hour']
        ss=spdf.head(1)['hour']
        return amds,pmds,sr,ss


    @reactive.Calc    
    def get_year_sp_df():
    
        year_start='2023:01:01 00:00:01'
        print("In get_year_sp_df")
        lat, long = loc()
        print("---------- loc:", loc(), "----------")
        tzone=timezone()
        tz = pytz.timezone(tzone)
        datetime_obj = tz.localize(datetime.strptime(year_start, "%Y-%m-%d"), is_dst=None)
        ref_date = tz.localize(datetime.strptime(year_start, "%Y-%m-%d"), is_dst=None)
        year_sp_df= pd.DataFrame()
                
        i=0;
        while datetime_obj.year==2023:
            diff=datetime_obj-ref_date
            utc_time = datetime_obj.astimezone(pytz.utc)
            if diff.days+1>68 and diff.days+1<307:# daylight saving compensation
                utc_time=utc_time+ timedelta(hours=1)
            solar_positions=solarposition.get_solarposition(utc_time,lat,long)
            azim=solar_positions["azimuth"][0]
            elev=solar_positions["elevation"][0]
            zenith=solar_positions["zenith"][0]
            if elev>0:
                
                year_sp_df.at[i,'hour']=datetime_obj.hour+datetime_obj.minute/60
                year_sp_df.at[i,'day']=diff.days+1
                year_sp_df.at[i,'zenth']=zenith
                year_sp_df.at[i,'azim']=azim
                year_sp_df.at[i,'elevation']=elev
                i+=1
                           
            datetime_obj=datetime_obj+ timedelta(minutes=10)
        
        
    @reactive.Calc    
    def get_day_sp_df():
        print("In get_day_sp_df")
        start_date= str(input.date())
        lat, long = loc()
        print("---------- loc:", lat, long, "----------")
        tzone=timezone()
        tz = pytz.timezone(tzone)
        datetime_obj = tz.localize(datetime.strptime(start_date, "%Y-%m-%d"), is_dst=None)
        ref_date = tz.localize(datetime.strptime(start_date, "%Y-%m-%d"), is_dst=None)
        
        sp_df= pd.DataFrame()
        
        i=0;
        while datetime_obj.day==ref_date.day:
            diff=datetime_obj-ref_date
            utc_time = datetime_obj.astimezone(pytz.utc)
            # if diff.days+1>68 and diff.days+1<307:# daylight saving compensation
            #     utc_time=utc_time+ timedelta(hours=1)
            solar_positions=solarposition.get_solarposition(utc_time,lat,long)
            azim=solar_positions["azimuth"][0]
            elev=solar_positions["elevation"][0]
            zenith=solar_positions["zenith"][0]
            if elev>0:
                
                sp_df.at[i,'hour']=datetime_obj.hour+datetime_obj.minute/60
                sp_df.at[i,'zenth']=zenith
                sp_df.at[i,'azim']=azim
                sp_df.at[i,'elevation']=elev
                sp_df.at[i,'utc_time']=str(utc_time.hour)+':'+str(utc_time.minute)
                sp_df.at[i,'loc_time']=str(datetime_obj.hour)+':'+str(datetime_obj.minute)

                sp_df.at[i,'hsh']=zenith-(cam_HVD_FOV[input.FOV()][0]/2)
                sp_df.at[i,'hsv']=zenith-(cam_HVD_FOV[input.FOV()][1]/2)
                sp_df.at[i,'hsd']=zenith-(cam_HVD_FOV[input.FOV()][2]/2)
                sp_df.at[i,'dsh']=elev-(cam_HVD_FOV[input.FOV()][0]/2)
                sp_df.at[i,'dsv']=elev-(cam_HVD_FOV[input.FOV()][1]/2)
                sp_df.at[i,'dsd']=elev-(cam_HVD_FOV[input.FOV()][2]/2)

                i+=1
                
            datetime_obj=datetime_obj+ timedelta(minutes=10)
            
        return sp_df
    
    @reactive.Calc
    def get_weather_data():
        print("In get_weather_data")
        weather_data = None
        # Find the timezone based on latitude and longitude
        timezone_data = pytz.timezone(timezone())

        print("input.date(): ", input.date())
        selected_date = input.date()
        print("selected_date: ", selected_date)

        lat, long = loc()
        today = date.today()
        today_time = datetime.today()
        print("today: ", today)
        print("today_time: ", today_time)
        date_difference = (selected_date - today).days
        print("date_difference: ", date_difference)

        if(date_difference < 0):
            weather_data = "No historical data to be shown"
        elif(0 <= date_difference < 5):
            datetime_start = timezone_data.localize(datetime.combine(selected_date, time.min))
            print("datetime_start: ", datetime_start.strftime("%Y-%m-%dT%H:%M:%S%z"))
            datetime_end = timezone_data.localize(datetime.combine(selected_date, time.max)) - timedelta(microseconds=1)
            print("datetime_end: ", datetime_end.strftime("%Y-%m-%dT%H:%M:%S%z"))
            url = "https://api.tomorrow.io/v4/timelines?apikey=Purg6j6hjn9LdzMVwRvToPbJVhnlSjAP"
            payload = {
                "location": f"{lat}, {long}",
                "fields": ["temperature", "windSpeed", "precipitationProbability", "cloudCover"],
                "units": "metric",
                "timesteps": ["1h"],
                "startTime": datetime_start.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "endTime": datetime_end.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "timezone": "auto"
            }
            headers = {
                "accept": "application/json",
                "Accept-Encoding": "gzip",
                "content-type": "application/json"
            }

            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Error fetching {url}: {response.status_code}")
            weather_data = response.json()
            # hourly_data = api_data['timelines']['hourly']
            # weather_data = pd.DataFrame([{**{'time': item['time']}, **item['values']} for item in hourly_data])
            # weather_data.fillna(0, inplace=True)

            # # Using convert_to_local_time to process the column 'time' data into local datetime string
            # weather_data['time'] = weather_data['time'].apply(convert_to_local_time)
            # weather_data['datetime'] = weather_data['time'].copy()
            # weather_data.set_index('datetime', inplace=True)
        else :
            weather_data = "Exceeding the forecast limit, up to four days can be predicted"
        
        print("----------------weather_data start----------------")
        print(weather_data)
        print("----------------weather_data end----------------")
        return weather_data
            
    @reactive.Calc
    def obj_names() -> List[str]:
        """Returns a split and *slightly* cleaned-up list of object names"""
        req(input.objects())
        return [x.strip() for x in input.objects().split(",") if x.strip() != ""]

    @reactive.Calc
    def obj_coords() -> List[SkyCoord]:
        return [SkyCoord.from_name(name) for name in obj_names()]

    @reactive.Calc
    def times_utc() -> Tuple[datetime, datetime]:
        req(input.date())
        lat, long = loc()
        sun = suntime.Sun(lat, long)
        return (
            sun.get_sunrise_time(input.date()),
            sun.get_sunset_time(input.date()),
        )

    @reactive.Calc
    def timezone() -> Optional[str]:
        lat, long = loc()
        return timezonefinder.TimezoneFinder().timezone_at(lat=lat, lng=long)

    @reactive.Calc
    def times_at_loc():
        start, end = times_utc()
        tz = pytz.timezone(timezone())
        return (start.astimezone(tz), end.astimezone(tz))

    @reactive.Calc
    def df() -> Dict[str, pd.DataFrame]:
        start, end = times_at_loc()
        times = pd.date_range(
            start - time_padding,
            end + time_padding,
            periods=100,
        )
        lat, long = loc()
        eloc = EarthLocation(lat=lat * u.deg, lon=long * u.deg, height=0)
        altaz_list = [
            obj.transform_to(AltAz(obstime=times, location=eloc))
            for obj in obj_coords()
        ]
        return {
            obj: pd.DataFrame(
                {
                    "obj": obj,
                    "time": times,
                    "alt": altaz.alt,
                    # Filter out discontinuity
                    "secz": np.where(altaz.alt > 0, altaz.secz, np.nan),
                }
            )
            for (altaz, obj) in zip(altaz_list, obj_names())
        }

    @output
    @render.plot
    def plot():
        fig, ax1 = plt.subplots(nrows=1)

        sunset, sunrise = times_at_loc()

        def add_boundary(ax, xval):
            ax.axvline(x=xval, c="#888888", ls="dashed")

        ax1.set_ylabel("Altitude (deg)")
        # ax1.set_xlabel("Time (Hour)")
        ax1.set_ylim(-5, 95)
        # ax1.set_xlim(sunset - time_padding, sunrise + time_padding)
        ax1.grid()
        # add_boundary(ax1, sunset)
        # add_boundary(ax1, sunrise)
        spdf=get_day_sp_df()
        print("spdf: ", spdf)
        ax1.scatter(spdf['hour'],spdf['elevation'],c=spdf['elevation'])
        # #diognal hotspot
        # hsd_zone=[x for x, val in enumerate(spdf['hsd']) if val <0]
        # if len(hsd_zone)>0:
        #     hsd_st=spdf['hour'][hsd_zone[0]]
        #     hsd_end=spdf['hour'][hsd_zone[-1]]
        #     ax1.axvline(x=hsd_st,  alpha=0.1, color='red')
        #     ax1.axvline(x=hsd_end,  alpha=0.1, color='red')
        #     ax1.axvspan(hsd_st, hsd_end, alpha=0.1, color='red')
            
        # horizontal hotspot
        hsh_zone=[x for x, val in enumerate(spdf['hsh']) if val <0]
        # print("hsh_zone: ", hsh_zone)
        ds=spdf.loc[spdf['dsh']>0]
        # print("ds: ", ds)
        amds=float(ds.head(1)['hour']) if len(ds)>0 else np.nan
        # print("amds: ", amds)
        pmds=float(ds.tail(1)['hour']) if len(ds)>0 else np.nan
        # print("pmds: ", pmds)
        if len(hsh_zone)>0:
            hsh_st=spdf['hour'][hsh_zone[0]]
            hsh_end=spdf['hour'][hsh_zone[-1]]
            ax1.axvline(x=hsh_st,  alpha=0.1, color='red')
            ax1.axvline(x=hsh_end,  alpha=0.1, color='red')
            ax1.axvspan(hsh_st, hsh_end, alpha=0.3, color='red')
            
            dsh_st=amds
            dsh_end=pmds
            sr=float(spdf.head(1)['hour']) if len(ds)>0 else np.nan
            ss=float(spdf.tail(1)['hour']) if len(ds)>0 else np.nan
            
            ax1.axvline(x=dsh_st,  alpha=0.3, color='gray')
            ax1.axvline(x=dsh_end,  alpha=0.3, color='gray')
            ax1.axvspan(sr,dsh_st, alpha=0.3, color='gray')
            ax1.axvspan(ss, dsh_end, alpha=0.3, color='gray')
            
            #sweet spot
            ax1.axvspan(dsh_st, hsh_st, alpha=0.5, color='springgreen')
            ax1.axvspan(hsh_end, dsh_end, alpha=0.5, color='springgreen')
            
        # # vertical hotspot
        # hsv_zone=[x for x, val in enumerate(spdf['hsv']) if val <0]
        # if len(hsv_zone)>0:
        #     hsv_st=spdf['hour'][hsv_zone[0]]
        #     hsv_end=spdf['hour'][hsv_zone[-1]]
        #     ax1.axvline(x=hsv_st,  alpha=0.1, color='red')
        #     ax1.axvline(x=hsv_end,  alpha=0.1, color='red')
        #     ax1.axvspan(hsv_st, hsv_end, alpha=0.1, color='red')
            
        # ax1.xaxis.set_major_locator(mpldates.AutoDateLocator())
        # ax1.xaxis.set_major_formatter(
        #     mpldates.DateFormatter("%H:%M", tz=pytz.timezone(timezone()))
        # )
        # ax1.legend(loc="upper right")

        # ax2.set_ylabel("Time of the day")
        # ax2.set_xlabel("Day of the year")
        # # ax2.set_ylim(4, 1)
        # # ax2.set_xlim(sunset - time_padding, sunrise + time_padding)
        # # ax2.grid()
        # # add_boundary(ax2, sunset)
        # # add_boundary(ax2, sunrise)
        # ax2.plot(spdf['hour'],spdf['elevation'])
        # ax2.xaxis.set_major_locator(mpldates.AutoDateLocator())
        # ax2.xaxis.set_major_formatter(
        #     mpldates.DateFormatter("%H:%M", tz=pytz.timezone(timezone()))
        # )

        return fig
    
    # Custom function to blend two colors
    def blend_colors(color1, color2, alpha=0.5):
        return (color1[0] * alpha + color2[0] * (1 - alpha),
                color1[1] * alpha + color2[1] * (1 - alpha),
                color1[2] * alpha + color2[2] * (1 - alpha),
                1.0)

    def plot_line_chart(times, weather_data, temperatures, wind_speeds):
        fig, ax1 = plt.subplots()
        # Plot Temperature data on the first axis
        ax1.plot(times, temperatures, label='Temperature (℃)', color='red')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Temperature (℃)', color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        # Create a second axis for Wind Speed
        ax2 = ax1.twinx()
        ax2.plot(times, wind_speeds, label='Wind Speed (m/s)', color='blue')
        ax2.set_ylabel('Wind Speed (m/s)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Set the x-axis date format
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))

        # Set the legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # shade the area
        for interval in weather_data['data']['timelines'][0]['intervals']:
            blue_color = mcolors.to_rgba('lightblue', alpha=0.5)
            yellow_color = mcolors.to_rgba('lightyellow', alpha=0.5)
            green_color = blend_colors(blue_color, yellow_color)

            start_time = datetime.fromisoformat(interval['startTime'][:-6])
            if interval['values']['precipitationProbability'] > 0:
                rect = Rectangle(
                    (mdates.date2num(start_time), ax1.get_ylim()[0]),
                    1 / 24,
                    ax1.get_ylim()[1] - ax1.get_ylim()[0],
                    facecolor=blue_color,
                    alpha=1.0  # Make the rectangle fully opaque
                )
                ax1.add_patch(rect)

            if interval['values']['cloudCover'] > 25:
                rect = Rectangle(
                    (mdates.date2num(start_time), ax1.get_ylim()[0]),
                    1 / 24,
                    ax1.get_ylim()[1] - ax1.get_ylim()[0],
                    facecolor=yellow_color,
                    alpha=1.0  # Make the rectangle fully opaque
                )
                ax1.add_patch(rect)

            # Custom blending for overlapping regions
            if interval['values']['precipitationProbability'] > 0 and interval['values']['cloudCover'] > 25:
                rect = Rectangle(
                    (mdates.date2num(start_time), ax1.get_ylim()[0]),
                    1 / 24,
                    ax1.get_ylim()[1] - ax1.get_ylim()[0],
                    facecolor=green_color,
                    alpha=1.0  # Make the rectangle fully opaque
                )
                ax1.add_patch(rect)

        plt.title('Weather Data')
        plt.grid(True)
        plt.tight_layout()
        return fig

    # process weather data
    def process_data(weather_data):
        times = []
        temperatures = []
        wind_speeds = []

        for interval in weather_data['data']['timelines'][0]['intervals']:
            time_str = interval['startTime'][:-6]
            time = datetime.fromisoformat(time_str)
            times.append(time)
            temperatures.append(interval['values']['temperature'])
            wind_speeds.append(interval['values']['windSpeed'])

        return np.array(times), np.array(temperatures), np.array(wind_speeds)
        
    @output
    @render.plot
    def weather_plot():
        weather_data = get_weather_data()
        if type(weather_data) is str:
            raise CustomError(weather_data)
        times, temperatures, wind_speeds = process_data(weather_data)
        fig = plot_line_chart(times, weather_data, temperatures, wind_speeds)
        return fig

    @output
    @render.table
    def table() -> pd.DataFrame:
        return get_day_sp_df()

    @output
    @render.ui
    def timeinfo():
            
        lat, long = loc()
        s = pd.DatetimeIndex([input.date()])
        localtz = pytz.timezone(timezone())
        pda=s.tz_localize(localtz)
        
        site_location=pvlib.location.Location(lat, long, tz=localtz)
        sunrise_set_transit=site_location.get_sun_rise_set_transit(pda)
        
        solarnoon_hr = sunrise_set_transit['transit'].dt.hour.values[0]
        solarnoon_mn = sunrise_set_transit['transit'].dt.minute.values[0]
        snhr=str(solarnoon_hr).zfill(2)
        snmn=str(solarnoon_mn).zfill(2)
        
        
        start_utc, end_utc = times_utc()
        start_at_loc, end_at_loc = times_at_loc()
        return ui.TagList(
            f"Sunrise: {start_at_loc.strftime('%H:%M')} ",
            f"Sunset: {end_at_loc.strftime('%H:%M')}, ",
            f"({timezone()})",
            
            ui.tags.br(),
            f"Sunrise: {start_utc.strftime('%H:%M')} ",
            f"Sunset: {end_utc.strftime('%H:%M')}, ",
            "(UTC)",
            
            ui.tags.br(),
            "Solarnoon:", snhr,":",snmn,
            f"({timezone()})",
            
            
        )
        # return "hi"

    # @output
    # @render.image
    # def im_plots():
    #     w=int(cam_XYratio[input.FOV()][0]/2)
    #     h=int(cam_XYratio[input.FOV()][1]/2)
    #     # img=np.random.randint(0, 255, size=(w, h, 1)).astype(np.uint8)
    #     A,B,C,D=[0.0217, 0.000106, 50.7, 27.6]
    #     E,F=100,100
    #     exc=100
    #     x = np.arange(-exc, w-exc, 1)
    #     y = np.arange(-exc, h-exc, 1)
    #     xx, yy = np.meshgrid(x, y, sparse=True)
    #     z2=B*(np.exp(-np.power((xx /E),2)*.5)* np.exp(-np.power((yy /F),2)*.5))
    #     img=255*(z2-np.min(z2))/(np.max(z2))
    #     cv2.imwrite("filename2.png", img)
    #     # im = Image.fromarray(img)
    #     # im.save("your_file.png")
    #     imgfile={'src':r"filename2.png",
    #          'width':w,'height':h}
        
    #     return imgfile


# The debug=True causes it to print messages to the console.
app = App(app_ui, server, debug=False)