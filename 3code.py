import requests
import json
import firebase_admin
from firebase_admin import credentials, messaging

# Constants
API_KEY = 'your_openweathermap_api_key'
CITY_ID = 'your_city_id'  # You can find this ID on OpenWeatherMap
TEMP_THRESHOLD = 35  # Temperature threshold for heat wave in Celsius

# Initialize Firebase Admin SDK
cred = credentials.Certificate('path_to_serviceAccountKey.json')
firebase_admin.initialize_app(cred)

def get_weather_data(city_id, api_key):
    url = f'http://api.openweathermap.org/data/2.5/weather?id={city_id}&appid={api_key}&units=metric'
    response = requests.get(url)
    return response.json()

def send_alert(temp, city_name):
    message = messaging.Message(
        notification=messaging.Notification(
            title='Heat Wave Alert!',
            body=f'Temperature in {city_name} is {temp}°C. Stay cool and hydrated!'
        ),
        topic='heatwaves'
    )
    response = messaging.send(message)
    print(f'Successfully sent message: {response}')

def check_heat_wave():
    weather_data = get_weather_data(CITY_ID, API_KEY)
    temp = weather_data['main']['temp']
    city_name = weather_data['name']
   
    if temp >= TEMP_THRESHOLD:
        send_alert(temp, city_name)

if __name__ == '__main__':
    check_heat_wave()FirebaseMessaging.getInstance().subscribeToTopic("heatwaves")
    .addOnCompleteListener(task -> {
        String msg = "Subscribed to heatwaves topic!";
        if (!task.isSuccessful()) {
            msg = "Subscription to heatwaves topic failed.";
        }
        Log.d(TAG, msg);
        Toast.makeText(MainActivity.this, msg, Toast.LENGTH_SHORT).show();
    });@Override
public void onMessageReceived(RemoteMessage remoteMessage) {
    // Handle FCM messages here.
    // Not getting messages here? See why this may be: https://goo.gl/39bRNJ
    Log.d(TAG, "From: " + remoteMessage.getFrom());

    // Check if message contains a notification payload.
    if (remoteMessage.getNotification() != null) {
        Log.d(TAG, "Message Notification Body: " + remoteMessage.getNotification().getBody());
        // Show notification to user
    }
}

