Among Us Predictor
===

The goal of this project is to come up with a model which will be able to predict whether crewmates or impostors should have won in a game of Among Us. It's not intended to be used predictively before a match, as at that time the roles of players aren't known. The idea is to eventually use it in the construction of a rating system, so the user would be able to generate a leaderboard of the players in their group for whatever roles or game settings.

The current state of the app is leading towards having the server maintain a model and all requests are handled by the server. This is rather sub-optimal.  
Eventually I think it would be good to have it such that the user could add their games, and a new model will be created. That model could be trained on the user's machine, then packaged into a zip file which they could save somewhere. Then later on when the user wants to see the leaderboards or add new games they would select that zipped model in the form.  
There may be a way to do this with python. But given my familiarity with tools like react, a javascript solution is sitting clearer in my mind. I know tensorflow has a javascript library, though I haven't investigated yet whether it's available as a front-end library or if it's even reasonable to expect something like that to be run by the user.

With all that being said, it's also possible that having this be more of a private instance where the user is running the 'server' on their own as well is quite probably a perfectly valid design choice.

---

## Data Format

Format input data as follows  
```js
[
    {
        crewmates: [
            "List",
            "Of",
            "Crewmates"
        ],
        impostors: [
            "Impostor",
            "List"
        ],
        map: "Airship",                 // "The Skeld" | "MIRA HQ" | "Polus" | "Airship"
        confirm_ejects: false,
        emergency_meetings: 1,
        emergency_cooldown: 20,
        discussion_time: 15,
        voting_time: 120,
        anonymous_votes: true,
        player_speed: 1.25,
        crewmate_vision: 0.75,
        impostor_vision: 1.5,
        kill_cooldown: 35.0,
        kill_distance: "Short",         // "Short" | "Medium" | "Long"
        visual_tasks: false,
        task_bar_updates: "Meetings",   // "Always" | "Meetings" | "Never"
        common_tasks: 2,
        long_tasks: 3,
        short_tasks: 5,
        match_winner: "crewmates"       // "crewmates" | "impostors"
    },
    { ... }
]
```

This doesn't apply to the html part, just the jupyter notebooks when loading from json. The flask side of things takes form data as of this moment.