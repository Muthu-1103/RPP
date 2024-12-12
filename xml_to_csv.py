import xml.etree.ElementTree as ET
import csv
import os
import glob

# Specify the folder containing the XML files
input_folder = 'E:/SEM8/Datasets/Ohio T1DM/OhioT1DM/2020/test/'
output_folder = 'E:/csv_output_test/'  # Folder where CSV files will be saved

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through all XML files in the folder
for xml_file in glob.glob(os.path.join(input_folder, '*.xml')):
    # Parse each XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract the patient ID to use it in the CSV file name (or use the file name itself)
    patient_id = root.attrib['id']
    xml_filename = os.path.basename(xml_file).split('.')[0]  # Get the base file name without extension

    # Construct the output path for the current CSV file
    output_path = os.path.join(output_folder, f'{xml_filename}_patient_{patient_id}.csv')

    # Open a CSV file for writing
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write CSV header with extended features
        headers = [
            'Patient ID', 'Weight', 'Insulin Type', 'Timestamp', 'Glucose Level', 'Finger Stick',
            'Basal Rate', 'Temp Basal', 'Bolus Dose', 'Meal Type', 'Carbs', 'Sleep Quality', 'Work Intensity',
            'Stressor Time', 'Hypo Event Time', 'Illness Time', 'Exercise Duration', 'Exercise Intensity',
            'Heart Rate', 'GSR', 'Skin Temp', 'Air Temp', 'Steps', 'Sleep Quality Sensor', 'Acceleration'
        ]
        writer.writerow(headers)

        # Extract patient information
        weight = root.attrib['weight']
        insulin_type = root.attrib['insulin_type']

        # Iterate through each event type
        for event in root.findall('.//glucose_level/event'):
            timestamp = event.attrib.get('ts', 'N/A')
            glucose_value = event.attrib.get('value', 'N/A')

            # Extract additional features
            finger_stick_event = root.find('.//finger_stick/event')
            finger_stick = finger_stick_event.attrib.get('value', 'N/A') if finger_stick_event is not None else 'N/A'

            basal_event = root.find('.//basal/event')
            basal_rate = basal_event.attrib.get('value', 'N/A') if basal_event is not None else 'N/A'

            temp_basal_event = root.find('.//temp_basal/event')
            temp_basal_rate = temp_basal_event.attrib.get('value', 'N/A') if temp_basal_event is not None else 'N/A'

            bolus_event = root.find('.//bolus/event')
            bolus_dose = bolus_event.attrib.get('dose', 'N/A') if bolus_event is not None else 'N/A'

            meal_event = root.find('.//meal/event')
            meal_type = meal_event.attrib.get('type', 'N/A') if meal_event is not None else 'N/A'
            meal_carbs = meal_event.attrib.get('carbs', 'N/A') if meal_event is not None else 'N/A'

            sleep_event = root.find('.//sleep/event')
            sleep_quality = sleep_event.attrib.get('quality', 'N/A') if sleep_event is not None else 'N/A'

            work_event = root.find('.//work/event')
            work_intensity = work_event.attrib.get('intensity', 'N/A') if work_event is not None else 'N/A'

            stressor_event = root.find('.//stressors/event')
            stressor_time = stressor_event.attrib.get('ts', 'N/A') if stressor_event is not None else 'N/A'

            hypo_event = root.find('.//hypo_event/event')
            hypo_time = hypo_event.attrib.get('ts', 'N/A') if hypo_event is not None else 'N/A'

            illness_event = root.find('.//illness/event')
            illness_time = illness_event.attrib.get('ts', 'N/A') if illness_event is not None else 'N/A'

            exercise_event = root.find('.//exercise/event')
            exercise_duration = exercise_event.attrib.get('duration', 'N/A') if exercise_event is not None else 'N/A'
            exercise_intensity = exercise_event.attrib.get('intensity', 'N/A') if exercise_event is not None else 'N/A'

            heart_rate_event = root.find('.//basis_heart_rate/event')
            heart_rate = heart_rate_event.attrib.get('value', 'N/A') if heart_rate_event is not None else 'N/A'

            gsr_event = root.find('.//basis_gsr/event')
            gsr_value = gsr_event.attrib.get('value', 'N/A') if gsr_event is not None else 'N/A'

            skin_temp_event = root.find('.//basis_skin_temperature/event')
            skin_temp = skin_temp_event.attrib.get('value', 'N/A') if skin_temp_event is not None else 'N/A'

            air_temp_event = root.find('.//basis_air_temperature/event')
            air_temp = air_temp_event.attrib.get('value', 'N/A') if air_temp_event is not None else 'N/A'

            steps_event = root.find('.//basis_steps/event')
            steps = steps_event.attrib.get('value', 'N/A') if steps_event is not None else 'N/A'

            sleep_sensor_event = root.find('.//basis_sleep/event')
            sleep_sensor_quality = sleep_sensor_event.attrib.get('quality', 'N/A') if sleep_sensor_event is not None else 'N/A'

            acceleration_event = root.find('.//acceleration/event')
            acceleration = acceleration_event.attrib.get('magnitude', 'N/A') if acceleration_event is not None else 'N/A'

            # Write each row to the CSV file
            writer.writerow([
                patient_id, weight, insulin_type, timestamp, glucose_value, finger_stick, basal_rate, temp_basal_rate,
                bolus_dose, meal_type, meal_carbs, sleep_quality, work_intensity, stressor_time, hypo_time, illness_time,
                exercise_duration, exercise_intensity, heart_rate, gsr_value, skin_temp, air_temp, steps, sleep_sensor_quality,
                acceleration
            ])

    print(f"CSV file saved successfully at {output_path}")
