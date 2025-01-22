"# API_online" 

Step1 : Install each and every library in requirements.txt 

Step2 : Place a .env file in the api folder to have the SQL Database credentials.

Step3: Run api.py and note the port that is being used

Step4: run ngrok.exe with the following line : ngrok http 6000  (6000 is an example of a port, the actual will be drawn from the api.py port.)

Step5: check the ngrok's url to see that the json works properly using the following as an example: https://9a3b-84-254-53-241.ngrok-free.app/api/get_routes_data?user_id=1

Step6: change the base_url in the app.py that is based on the one the ngrok has generated. example : https://9a3b-84-254-53-241.ngrok-free.app/api/get_routes_data


Local Host edition:

Step7: Run the app.py and enter after the local ip the user you want to see. example : http://127.0.0.1:7777/?user=2


Render Host edition: (requires to have Git installed on your Computer)

Step7: You create a new Repository on GitHub.

Step8: Copy the …or create a new repository on the command line

Step9: Go to the folder that contains app.py and requirements.txt and open a Windows PowerShell and paste the git lines.

Step10: Drag & Drop the app.py and requirements.txt into the new repository on GitHub.

Step11: https://dashboard.render.com/   -> New -> Web Service -> Public Git Repository -> and paste the https code from your depository example: https://github.com/dimitris986/testing_2.git

Step12: You fill the following blanks:
	Region: Frankfurt (EU Central)
	Build Command: pip install -r requirements.txt
	Start Command: gunicorn app:server  --app should be replaced from whatever your .py is named. For example if you have named it app_pycharm.py , then you will type gunicorn app_pycharm:server
	Free

Step13: Deploy Web Service

Step14: Render provides you with a link and then the only thing you have to do is to insert the user likes the following example: https://api-online-b3sk.onrender.com/?user=1

Important steps along the way:

i)That if you want to use Windows PowerShell to connect your PC Folder with the repository, you have to install it with PATH enabled.

ii)Make sure that your GitHub version of your app contains a line :  server = app.server

For resusing it , you just have to change the base_url when you create the ngrok link for your api.
