import gpxpy
import math
import folium
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')  # Set the backend
import matplotlib.colors as mcolors
import os
import numpy as np


# Helper function to calculate distance between two points
#using haversine distance
def distance(point1, point2):
    # Radius of the Earth in m
    lat1, lon1 = point1['lat'] if isinstance(point1,dict) and 'lat' in point1.keys() else point1.latitude, point1['lon'] if isinstance(point1,dict) and  'lon' in point1.keys() else point1.longitude
    lat2, lon2 = point2['lat'] if isinstance(point2,dict) and 'lat' in point2.keys() else point2.latitude, point2['lon'] if isinstance(point2,dict) and 'lon' in point2.keys() else point2.longitude
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = 1000 * R * c

    return distance


def plot_elevation_chart(points, raw_points):
    # Create a figure and a subplot
    plt.figure(figsize=(12, 6))
    fig, ax = plt.subplots()

    plt.plot(raw_points.index, raw_points['elev'])
    
    # Normalize the gradients for color mapping
    gradients = [point['gradient'] for point in points]
    norm = mcolors.Normalize(vmin=min(gradients), vmax=25)#max(gradients))
    colormap = plt.cm.jet

    # Plot each segment with a color based on its gradient
    for i in range(1, len(points)):
        ax.plot([points[i-1]['dist'], points[i]['dist']],
                [points[i-1]['elev'], points[i]['elev']],
                color=colormap(norm(points[i]['gradient'])),
                linewidth=2)

    # Setting the labels and title
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Elevation (m)')
    ax.set_title('Elevation Profile with Gradient Coloring')

    # Adding a colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # This is necessary for the colorbar to show up
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Gradient (%)')

    # Show the plot
    plt.ylim(0, 2000)
    plt.savefig('elevation_chart.png')
    plt.show()
    
def find_index(points, target_dist, default):
    """Find the index of the point where the distance is just greater than or equal to target_dist."""
    for i, point in enumerate(points):
        if point['dist'] >= target_dist:
            return i
    return default

def plot_elevation_chart_with_features(points, features):
    #plt.figure()
    fig, ax = plt.subplots()
    for feature in features:
        # Determine the color based on the feature type
        color = {'flat': 'green', 'climb': 'red', 'hill': 'blue', 'descent': 'yellow'}.get(feature['type'], 'gray')

        # Find the start and end indices for this feature
        start_index = find_index(points, feature['start_km'], 0)
        end_index = find_index(points, feature['end_km'], len(points) - 1)

        # Extract distances and elevations for this feature
        feature_points = points[start_index:end_index + 1]
        distances = [p['dist'] for p in feature_points]
        elevations = [p['elev'] for p in feature_points]

        # Plot this feature
        ax.plot(distances, elevations, color=color, linewidth=2)

    plt.xlabel('Distance (km)')
    plt.ylabel('Elevation (m)')
    plt.title('Elevation Profile with Feature Types')
    plt.grid(True)
    plt.ylim(min([p['elev'] if p['elev']!=None else 0 for p in points]), max([p['elev'] if p['elev']!=None else 0 for p in points])+200)
    
    # Set the aspect ratio
    ax.set_aspect(40)
    
    plt.savefig('elevation_chart.png')
    plt.show()


def plot_map(points, output_html='map_output.html'):
    # Create the map centered around the average coordinates
    average_lat = sum(point['lat'] for point in points) / len(points)
    average_lon = sum(point['lon'] for point in points) / len(points)
    map = folium.Map(location=[average_lat, average_lon], zoom_start=12)

    # Prepare gradient colors
    gradients = [point['gradient'] for point in points]
    norm = mcolors.Normalize(vmin=min(gradients), vmax=25)#max(gradients))
    colormap = plt.cm.jet

    # Add lines to the map
    for i in range(1, len(points)):
        color = mcolors.to_hex(colormap(norm(points[i]['gradient'])))
        folium.PolyLine(
            locations=[[points[i-1]['lat'], points[i-1]['lon']], [points[i]['lat'], points[i]['lon']]],
            color=color,
            weight=5
        ).add_to(map)

    # Save the map to an HTML file
    map.save(output_html)
    print(f'Map saved as {output_html}')

def plot_folium_map_with_features(points, features, output_html='map_output.html'):
    # Create the map centered around the average coordinates
    average_lat = sum(point['lat'] for point in points) / len(points)
    average_lon = sum(point['lon'] for point in points) / len(points)
    map = folium.Map(location=[average_lat, average_lon], zoom_start=12)
    
    start_point = points[0]
    
    # Add custom markers
    folium.Marker(
        location=[start_point['lat'], start_point['lon']],
        icon=folium.Icon(icon='play', color='green'),
        popup='Start'
    ).add_to(map)

    end_point = points[-1]
    folium.Marker(
        location=[end_point['lat'], end_point['lon']],
        icon=folium.Icon(icon='flag', color='red'),
        popup='Finish'
    ).add_to(map)

    feature_colors = {
        'flat': 'green',
        'climb': 'red',
        'hill': 'blue',
        'descent': 'yellow'
    }

    for feature in features:
        # Determine the color based on the feature type
        color = feature_colors.get(feature['type'], 'gray')

        # Find the start and end indices for this feature
        start_index = next((i for i, p in enumerate(points) if p['dist'] >= feature['start_km']), 0)
        end_index = next((i for i, p in enumerate(points) if p['dist'] >= feature['end_km']), len(points) - 1)

        # Extract the latitudes and longitudes for this feature
        latitudes = [points[i]['lat'] for i in range(start_index, end_index + 1)]
        longitudes = [points[i]['lon'] for i in range(start_index, end_index + 1)]

        # Add a polyline to the map
        folium.PolyLine(list(zip(latitudes, longitudes)), color=color, weight=5).add_to(map)

    # Save the map to an HTML file
    map.save(output_html)


def parse_gpx(file_path, min_distance=250, min_precise_distance=25):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        regularized_points = []
        points = []
        total_distance = 0
        total_distance2 = 0
        accumulated_distance = 0
        accumulated_elevation = 0
        last_point = None

        i=-1
        for track in gpx.tracks:
            for segment in track.segments:
                elevations = []
                gradients = []
                distances = []
                indices = []
                segment_distance=0
                
                for point in segment.points: 
                    i+=1
                    indices.append(i)
                    if last_point is not None:
                        segment_distance = distance(last_point, point)
                        if segment_distance == 0 :
                            continue
                        total_distance2 += segment_distance
                        accumulated_distance += segment_distance
                        elevation_gain = (point.elevation if point.elevation != None else 0) - (last_point.elevation if last_point.elevation != None else 0)
                        accumulated_elevation += elevation_gain
                        gradient = (elevation_gain / segment_distance) * 100 if segment_distance > 0 else 0

                        if segment_distance <= min_precise_distance:
                            points.append({
                                'index': i,
                                'lat': point.latitude,
                                'lon' : point.longitude,
                                'elev': point.elevation,
                                'dist': total_distance2,
                                'segment_dist': segment_distance
                            })
                            last_point = point
                            continue
                        
                        gradients.append(gradient)
                        distances.append(segment_distance)
                        elevations.append(elevation_gain)

                        if accumulated_distance >= min_distance:
                            #print((i,accumulated_distance))
                            total_distance = total_distance + accumulated_distance
                            accumulated_gradient = (accumulated_elevation / accumulated_distance) * 100
                            regularized_points.append({
                                'lat': point.latitude, 
                                'lon': point.longitude, 
                                'elev': point.elevation, 
                                'delta_dist': accumulated_distance,
                                'dist': total_distance,
                                'gradient': accumulated_gradient,
                                'max_gradient': np.max(np.append(np.array(gradients),accumulated_gradient)),
                                'min_gradient' : np.min(np.array(gradients)),
                                'indices': indices,
                                'feature_type': ''  # To be updated later
                            })
                            accumulated_distance = 0
                            accumulated_elevation = 0
                            if np.max(np.array(gradients)) > 25:
                                print({'dist':distances,'elev':elevations,'grad':gradients})
                            gradients = []
                            distances= []
                            elevations= []
                            indices = []
                    points.append({
                        'index': i,
                        'lat': point.latitude,
                        'lon' : point.longitude,
                        'elev': point.elevation,
                        'dist': total_distance2,
                        'segment_dist': segment_distance
                    })
                    last_point = point
                        

        return regularized_points, total_distance, points

def classify_point(gradient, max_gradient, min_gradient, climb_length, is_cobble, is_last_point):
    if gradient < 5 and is_last_point:  # Si último punto = si Y gradiente medio < 5, Sprint
        return 'Sprint'
    elif gradient < -3:
        return 'Downhill'
    elif gradient >= 3:
        if climb_length < 2000:
            if gradient < 5:
                if min_gradient > -3:
                    return 'Flat Hills Cobblestone ND' if is_cobble else 'Hills Flat ND'
                else:
                    return 'Flat Hills Cobblestone' if is_cobble else 'Hills Flat'
            else:
                if min_gradient > -3:
                    return 'Hills Cobblestone ND' if is_cobble else 'Hills ND'
                else:
                    return 'Hills Cobblestone' if is_cobble else 'Hills'
        elif climb_length < 3000:
            if gradient > 10:
                if min_gradient > -3:
                    return 'Cobblestone Hills Climbing ND' if is_cobble else 'Hills Climbing ND'
                else:
                    return 'Cobblestone Hills Climbing' if is_cobble else 'Hills Climbing'
            else:
                if min_gradient > -3:
                    return 'Hills Cobblestone ND' if is_cobble else 'Hills ND'
                else:
                    return 'Hills Cobblestone' if is_cobble else 'Hills'
        elif climb_length < 5000:
            if gradient > 8:
                if min_gradient > -3:
                    return 'Cobblestone Hills Climbing ND' if is_cobble else 'Climbing Hills ND'
                else:
                    return 'Cobblestone Hills Climbing' if is_cobble else 'Climbing Hills'
            elif gradient > 5:
                if min_gradient > -3:
                    return 'Cobblestone Hills Climbing ND' if is_cobble else 'Hills Climbing ND'
                else:
                    return 'Cobblestone Hills Climbing' if is_cobble else 'Hills Climbing'
            else:
                if min_gradient > -3:
                    return 'Hills Cobblestone ND' if is_cobble else 'Hills ND'
                else:
                    return 'Hills Cobblestone' if is_cobble else 'Hills'
        elif climb_length < 8000:
            if gradient > 8:
                if min_gradient > -3:
                    return 'Cobblestone Climbing ND' if is_cobble else 'Climbing ND'
                else:
                    return 'Cobblestone Climbing' if is_cobble else 'Climbing'
            elif gradient > 5:
                if min_gradient > -3:
                    return 'Cobblestone Hills Climbing ND' if is_cobble else 'Climbing Hills ND'
                else:
                    return 'Cobblestone Hills Climbing' if is_cobble else 'Climbing Hills'
            else:
                if min_gradient > -3:
                    return 'Cobblestone Hills Climbing ND' if is_cobble else 'Hills Climbing ND'
                else:
                    return 'Cobblestone Hills Climbing' if is_cobble else 'Hills Climbing'
        else:
            if gradient < 5:
                if min_gradient > -3:
                    return 'Cobblestone Flat Climbing' if is_cobble else 'Flat Climbing'
                else:
                    return 'Cobblestone Hills Climbing' if is_cobble else 'Climbing Hills'
            else:
                if min_gradient > -3:
                    return 'Cobblestone Climbing ND' if is_cobble else 'Climbing ND'
                else:
                    return 'Cobblestone Climbing' if is_cobble else 'Climbing'
    else:
        if max_gradient < 5.5:
            return 'Flat Cobblestones' if is_cobble else 'Flat'
        else:
            if min_gradient < -3:
                return 'Flat Hills Cobblestone ND' if is_cobble else 'Flat Hills ND'
            else:
                return 'Flat Hills Cobblestone' if is_cobble else 'Flat Hills'

'''
def merge_and_reclassify_sections(sections):
    merged_sections = []
    current_section = sections[0]

    for next_section in sections[1:]:
        # Merge if same type or if adjacent types can be combined
        if next_section['type'] == current_section['type'] or \
           (current_section['type'] in ['hill', 'potential_climb'] and next_section['type'] in ['hill', 'potential_climb']):
            current_section['length'] += next_section['length']
            current_section['end_km'] = next_section['end_km']
        else:
            # Reclassify if necessary
            if current_section['type'] == 'potential_climb':
                current_section['type'] = 'climb' if current_section['length'] >= 4000 else 'hill'
            merged_sections.append(current_section)
            current_section = next_section

    # Add last section
    if current_section['type'] == 'potential_climb':
        current_section['type'] = 'climb' if current_section['length'] >= 4000 else 'hill'
    merged_sections.append(current_section)

    return merged_sections
'''
#!CAPAR DIFICULTAD A 100 (NO LO HE AÑADIDO AQUI, QUIZAS CUANDO HAGAS LA LLAMADA A ESTA FUNCION?)
#Habra que llamar a esta funcion para asignar la dificultad de cada segmento. Creo que hay que definir el parametro dificultad para segmentos.
def calculate_difficulty(segment_type, climb_length, gradient, feature_length):
    difficulty = 0
    if segment_type == 'Flat':
        if gradient <= 0:
            difficulty = 0
            #return difficulty
        else:    
            difficulty = gradient/3*5
            #return difficulty
    elif segment_type == 'Flat Hills' or segment_type =='Flat Hills ND':
        if gradient <= 0:
            difficulty = 5
            #return difficulty
        else:
            difficulty = 5 + (gradient/3*5)
            #return difficulty
    elif segment_type == 'Hills Flat' or segment_type =='Hills Flat ND':
        difficulty = 10 + ((gradient-3)/2*5)
        #return difficulty
    elif segment_type == 'Hills' or segment_type =='Hills ND' or segment_type =='Hills Time Trial':
        difficulty = 15 + ((gradient-5)/5*10)
        #return difficulty
    elif segment_type == 'Hills Climbing' or segment_type =='Hills Climbing ND':
        difficulty = 25 + ((climb_length-2000)/5000*6.6) + ((gradient-3)/6*13.4)
        #return difficulty
    elif segment_type == 'Climbing Hills' or segment_type =='Climbing Hills ND' or segment_type =='Cobblestone Hills Climbing' or segment_type =='Cobblestone Hills Climbing ND' or segment_type =='Flat Climbing' or segment_type =='Cobblestone Flat Climbing' or segment_type =='Climbing Hills Time Trial':
        difficulty = 30 + ((climb_length-3000)/4000*11.55) + ((gradient-5)/5*23.45)
        #return difficulty
    elif segment_type == 'Climbing' or segment_type =='Climbing ND' or segment_type =='Cobblestone Climbing' or segment_type =='Cobblestone Climbing ND' or segment_type =='Climbing Time Trial':
        difficulty = 50 + ((climb_length-5000)/10000*16.5) + ((gradient-5)/6*33.5)
        #return difficulty
    elif segment_type == 'Sprint':
        difficulty = (climb_length/6000*20) + (gradient/5*15)
        #return difficulty
    elif segment_type == 'Flat Cobblestone' or segment_type =='Cobblestone Time Trial':
        difficulty = (feature_length/3000*30) + (gradient/3*5)
        #return difficulty
    elif segment_type == 'Flat Hills Cobblestone' or segment_type =='Flat Hills Cobblestone ND':
        difficulty = 5 + (feature_length/3000*25) + (gradient/3*5)
        #return difficulty
    elif segment_type == 'Hills Cobblestone' or segment_type =='Hills Cobblestone ND' or segment_type =='Hills Cobblestone Time Trial':
        difficulty = 15 + (feature_length/3000*5) + ((gradient-3)/3*8)
        #return difficulty
    elif segment_type == 'Downhill' or segment_type =='Downhill Time Trial':
        difficulty = ((-gradient-3)/2*5)
        #return difficulty
    elif segment_type == 'Flat Time Trial':
        if gradient <= 0:
            difficulty = 0
            #return difficulty
        else:
            difficulty = gradient/3*7.5
            #return difficulty
    elif segment_type == 'Flat Hills Time Trial':
        difficulty = 7.5 + ((gradient-3)/2*7.5)
        #return difficulty
    return difficulty
    
        



    
    # Calculate the average and maximum gradient over the feature length
    #start_index = next(i for i, p in enumerate(points) if p['dist'] >= feature['start_km'])
    #end_index = next(i for i, p in enumerate(points) if p['dist'] >= feature['end_km'])

    #gradients = [p['gradient'] for p in points[start_index:end_index]]
    #total_gradient = sum(gradients)
    #average_gradient = total_gradient / len(gradients) if len(gradients) > 0 else 0
    #max_gradient = max(gradients, default=0)

    # Calculate difficulty as a product of length and average gradient
    #difficulty = feature['length'] * abs(average_gradient)

    # Store average and maximum gradient in the feature
    #feature['average_gradient'] = average_gradient
    #feature['max_gradient'] = max_gradient

    #return difficulty


def calculate_climb_length(df, rest_threshold = 1200):
    climb_length = 0
    current_indices = []
    extra_already_processed = []
    
    for i in range(0, len(df)):
        if i in extra_already_processed:
            #print(f' skipping {i} because it was processed')
            continue
        row = df.iloc[i]        
        if row['gradient'] >= 3:
            if (i == len(df) -1) :
                current_indices.append(i)
                climb_length += row['delta_dist']
                for k in current_indices:
                    df.loc[k, 'climb_length'] = climb_length
            else:
                current_indices.append(i)
                climb_length += row['delta_dist']
        else:
            # looking for rest segments in climbs
            extra_dist = 0
            for k in range(1,20):
                if i+k >= len(df): 
                    break
                row_extra = df.iloc[i+k]
                if row_extra['gradient'] >=3 :
                    extra_dist = 0
                    #print(f'index: {i+k} continues the previous climb: {current_indices}')
                    extra_already_processed.append(i+k)
                    climb_length += row_extra['delta_dist']
                    current_indices.append(i+k)
                else:
                    #print(f'index: {i+k} may be a gap in a climb')
                    if extra_dist >= rest_threshold:
                        #print(f'finishing looking at gapsin climb here: {i+k}')
                        extra_dist += row_extra['delta_dist']
                        break
                    extra_dist += row_extra['delta_dist']
            for j in current_indices:
                #if j > 200:
                    #print(f'{j} climb length {climb_length}')
                df.loc[j, 'climb_length'] = climb_length if df.iloc[j]['climb_length'] == df.iloc[j]['delta_dist'] else df.iloc[j]['climb_length']
            current_indices = []
            climb_length = 0
            

def identify_features(points):

    points['climb_length'] = points['delta_dist']
    
    calculate_climb_length(points)
    
    # Classify aggregated segments and project back to original DataFrame
    segment_types = []
    
    for index,segment in points.iterrows():
        is_last_point = True if index == len(points) -1 else False
        segment_type = classify_point(segment['gradient'], segment['max_gradient'], segment['min_gradient'], segment['climb_length'], False, is_last_point)
        segment_types.append(segment_type)
    
        
    # Add classifications to the original DataFrame
    points['segment_type'] = segment_types

    return points


'''
def identify_features(points):
    sections = []
    current_section = {'type': classify_point(points[0]['gradient']), 'start_km': 0, 'end_km': 0, 'length': 0}

    for i in range(1, len(points)):
        point_type = classify_point(points[i]['gradient'])
        current_section['length'] += points[i]['dist'] - points[i-1]['dist']
        current_section['end_km'] = points[i]['dist']

        if point_type != current_section['type']:
            sections.append(current_section)
            current_section = {'type': point_type, 'start_km': points[i]['dist'], 'end_km': points[i]['dist'], 'length': 0}

    # Add the last section
    sections.append(current_section)

    # Merge and reclassify sections
    features = merge_and_reclassify_sections(sections)

    # Calculate difficulty and end proximity for each feature
    for feature in features:
        feature['difficulty'] = calculate_difficulty(feature, points)  # Implement this function based on your criteria
        feature['end_proximity'] = points[-1]['dist'] - feature['end_km']

    return features
'''
#Hay que definir un array de porcentajes para cada segmento, en algun lado arriba, para poner lo que salga de esta funcion
#Definir is_tt para cada etapa asi se puede usar aqui y en el motor, que hará falta
#Si no es muy dificil, hacer pruebas comprobando que la suma de porcentajes es 1
def calculate_segment_abilities(terrain_df, segment_type, distance, is_tt):
    total = 0
    abilities = {'Stamina': 0, 'Sprint': 0, 'Climbing': 0, 'Flat': 0, 'Technique': 0, 'Downhill': 0, 'Hills': 0, 'Aggressiveness': 0, 'Teamwork' : 0}
    abilities['Sprint'] = terrain_df[terrain_df['Type'] == segment_type]['Sprint'].values[0]
    abilities['Climbing'] = terrain_df[terrain_df['Type'] == segment_type]['Climbing'].values[0]
    abilities['Flat'] = terrain_df[terrain_df['Type'] == segment_type]['Flat'].values[0]
    abilities['Technique'] = terrain_df[terrain_df['Type'] == segment_type]['Technique'].values[0]
    abilities['Downhill'] = terrain_df[terrain_df['Type'] == segment_type]['Downhill'].values[0]
    abilities['Hills'] = terrain_df[terrain_df['Type'] == segment_type]['Hills'].values[0]
    abilities['Aggressiveness'] = terrain_df[terrain_df['Type'] == segment_type]['Aggressiveness'].values[0]
    abilities['Teamwork'] = terrain_df[terrain_df['Type'] == segment_type]['Teamwork'].values[0]
    if is_tt:
         abilities['Stamina'] = terrain_df[terrain_df['Type'] == segment_type]['Stamina'].values[0] * distance / 20000
    else:
         abilities['Stamina'] = terrain_df[terrain_df['Type'] == segment_type]['Stamina'].values[0] * distance / 120000
    total_no_stamina = abilities['Sprint'] + abilities['Climbing'] + abilities['Flat'] + abilities['Technique'] + abilities['Downhill'] + abilities['Hills'] + abilities['Aggressiveness'] + abilities['Teamwork']
    abilities['Sprint'] = abilities['Sprint'] * (1 - abilities['Stamina']) / total_no_stamina
    abilities['Climbing'] = abilities['Climbing'] * (1 - abilities['Stamina']) / total_no_stamina 
    abilities['Flat'] = abilities['Flat'] * (1 - abilities['Stamina']) / total_no_stamina 
    abilities['Technique'] = abilities['Technique'] * (1 - abilities['Stamina']) / total_no_stamina 
    abilities['Downhill'] = abilities['Downhill'] * (1 - abilities['Stamina']) / total_no_stamina 
    abilities['Hills'] = abilities['Hills'] * (1 - abilities['Stamina']) / total_no_stamina 
    abilities['Aggressiveness'] = abilities['Aggressiveness'] * (1 - abilities['Stamina']) / total_no_stamina 
    abilities['Teamwork'] = abilities['Teamwork'] * (1 - abilities['Stamina']) / total_no_stamina
    return abilities

#No he "borrado" esto aun, te lo dejo a ti
def calculate_abilities(features, total_distance):
    abilities = {'Stamina': 0, 'Sprint': 0, 'Climbing': 0, 'Flat': 0, 'Technique': 0, 'Downhill': 0, 'Hills': 0, 'Aggressiveness': 0, 'Teamwork' : 0}
  # Define maximum percentages for skill categories
    max_main_skill = 50  # For primary skills
    max_secondary_skill = 30  # For secondary skills
    max_tertiary_skill = 10  # For tertiary skills

    # Skill category mappings
    skill_categories = {
        'main': ['Climbing', 'Sprint', 'Hills'],
        'secondary': ['Stamina', 'Flat', 'Downhill'],
        'tertiary': ['Technique', 'Aggressiveness', 'Teamwork']
    }

    overall_difficulty = sum(feature['difficulty'] for feature in features) / len(features)
    stamina_base = 4 + (total_distance / 250) * 11  # Base value ranging from 4% to 15%
    abilities['Stamina'] = stamina_base + overall_difficulty

    # Iterate through each feature to calculate its influence on abilities
    for feature in features:
        # Calculate the proximity factor (closer to end has more influence)
        proximity_factor = (total_distance - feature['end_km']) / total_distance
        influence_factor = feature['length'] * feature['difficulty'] * (1 - proximity_factor)

        # Define how each feature type influences different abilities
        if feature['type'] == 'flat':
            abilities['Flat'] += influence_factor
            abilities['Sprint'] += influence_factor * 0.2
            abilities['Stamina'] += influence_factor * 0.2
            abilities['Teamwork'] += influence_factor * 0.3
        elif feature['type'] == 'hill':
            abilities['Hills'] += influence_factor
            abilities['Aggressiveness'] += influence_factor * 0.8
            abilities['Stamina'] += influence_factor * 0.5
        elif feature['type'] == 'climb':
            abilities['Climbing'] += influence_factor
            abilities['Stamina'] += influence_factor * 0.7
        elif feature['type'] == 'descent':
            abilities['Downhill'] += influence_factor
            abilities['Technique'] += influence_factor * 0.5
            abilities['Aggressiveness'] += influence_factor * 0.5


    # Adjust Stamina based on total race length and overall difficulty
    


    # Apply scaling factor to maintain ratio and respect caps
    for category, skills in skill_categories.items():
        max_cap = max_main_skill if category == 'main' else max_secondary_skill if category == 'secondary' else max_tertiary_skill
        total_in_category = sum(abilities[skill] for skill in skills)
        if total_in_category > max_cap:
            scale_factor = max_cap / total_in_category
            for skill in skills:
                abilities[skill] *= scale_factor


    # Normalize abilities to sum up to 100%
    total_ability_value = sum(abilities.values())
    if total_ability_value > 0:
        scale_factor = 100 / total_ability_value
        for ability in abilities:
            abilities[ability] = round(abilities[ability] * scale_factor, 2)

    return abilities


def main(file_path):
    # Recursively search for GPX files
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".gpx"):
                file_path = os.path.join(root, file)
                regularized_points, total_distance = parse_gpx(file_path, min_distance=500)
                features = identify_features(regularized_points)
                abilities = calculate_abilities(features, total_distance)
                
                
                stage_data = {
                    'points' : regularized_points,
                    'features' : features,
                    'abilities' : abilities,
                    'total_distance' : total_distance,
                    'name': file_path.split('/')[-3],
                    'year': file_path.split('/')[-2],
                    'stage': file.split('.')[-2],
                    'filepath': file_path
                }
                

if __name__ == "__main__":
    #directory_path = '/workspace/gpx_scraper/'  
    main(directory_path)


# Example Usage
# regularized_points, total_distance = parse_gpx("/workspace/gpx_scraper/tour-de-france/2024/stage-1-route.gpx", min_distance=500 )#parse_gpx("/workspace/gpx_scraper/amstel-gold-race/2022/route-men.gpx", min_distance=500)
# print(regularized_points)

# features = identify_features(regularized_points)
# print(features)


# plot_elevation_chart_with_features(regularized_points, features)
# plot_folium_map_with_features(regularized_points, features, 'my_stage_map.html')

# # Continue with feature identification and analysis using regularized_points
# total_distance = regularized_points[-1]['dist']
# abilities = calculate_abilities(features, total_distance)
# print(abilities)
