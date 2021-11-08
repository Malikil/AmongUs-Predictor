from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from custom_classes import Predictor

app = Flask(__name__)
model = Predictor.load('model')

def validate_data(form_data):
    data = {}
    keys = [
        ('player_count', int),
        ('impostor_count', int),
        ('map', str),
        ('confirm_ejects', bool),
        ('emergency_meetings', int),
        ('emergency_cooldown', int),
        ('discussion_time', int),
        ('voting_time', int),
        ('anonymous_votes', bool),
        ('player_speed', float),
        ('crewmate_vision', float),
        ('impostor_vision', float),
        ('kill_cooldown', float),
        ('kill_distance', str),
        ('visual_tasks', bool),
        ('task_bar_updates', str),
        ('common_tasks', int),
        ('long_tasks', int),
        ('short_tasks', int)
    ]
    for k in keys:
        data[k[0]] = k[1](form_data.get(k[0]))
    return data

@app.route('/')
def root():
    return render_template('leaders-form.html')

@app.route('/leaders', methods=['GET', 'POST'])
def leaders():
    try:
        jdata = validate_data(request.form)
    except TypeError:
        return redirect('/')
        
    # Get some leaders
    # This will be sloooooooooow, but I'll make it work first and
    # deal with speed later
    leaders = model.get_leaders(jdata, verbose=True)
    return render_template('leaders.html', leaders=leaders)

if __name__ == '__main__':
    app.run(debug=True)
