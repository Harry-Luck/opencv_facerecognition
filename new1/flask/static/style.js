// speech to text(using mic)
var SpeechRecognition = window.webkitSpeechRecognition;
var recognition = new SpeechRecognition();

var textbox = $(".chat-inputbox");
var content = "";

recognition.continuous = true;
recognition.onresult = function (event) {
    var current = event.resultIndex;
    var transcript = event.results[current][0].transcript;

    content += transcript;
    textbox.val(content);

    chatbox.onSendButton();
    
    content = "";
};

$(".mic-btn").on("click", function() {
    this.state = !this.state;

    if(this.state) {
        console.log('start');
        if (content.length) {
            content += " ";
        }
        recognition.start();
        $('.recording').addClass('active');
        $('.chat-form-message-input').addClass('hide');
        $('.chat-form-message').addClass('recoding-status');
    } else {
        console.log('stop');
        recognition.stop();
        $('.recording').removeClass('active');
        $('.chat-form-message-input').removeClass('hide');
        $('.chat-form-message').removeClass('recoding-status');
    }
});

textbox.on("input", function () {
    content = $(this).val();
});

class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chat-bot-avatar'),
            closeButton: document.querySelector('.control-overlay-collapse'),
            chatBox: document.querySelector('.chat-expanded'),
            chatInput: document.querySelector('.chat-inputbox'),
            chatMic: document.querySelector('.mic-btn'),
            sendButton: document.querySelector('.form-button')
        }

        this.state = false;
        this.messages = [];

        localStorage.setItem("jwt-token", "");

    }

    display() {

        // display default greeting of chatbot
        let msgDefaultTxt = "Hello, I'm Jad, the Entrepreneur Chatbot. How I can help you?"
        let msgDefault = { name: "Sam", message: msgDefaultTxt }
        this.messages.push(msgDefault);
        
        var defalutHtml = '<div class="chat-message-bubble chat-message-bubble--bot"><div class="chat-message-default"><p>' + msgDefaultTxt + '</p></div></div>'

        document.querySelector('.chat-history').innerHTML = defalutHtml;

        // event
        const {openButton, closeButton, chatBox, chatInput, chatMic, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(chatBox))
        closeButton.addEventListener('click', () => this.toggleState(chatBox))
        chatInput.addEventListener('focus', () => this.toggleButton(chatMic, sendButton))
        sendButton.addEventListener('click', () => this.onSendButton())

        chatInput.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton()
            }
        })
        
    }

    toggleState(chatbox) {
        this.state = !this.state;

        // show or hides the button
        if(this.state) {
            chatbox.classList.add('active')
        } else {
            chatbox.classList.remove('active')
        }
    }

    toggleButton(chatMic, sendButton) {
        sendButton.classList.add('active');        
        chatMic.classList.remove('active');
    }

    onSendButton() {
        const {chatInput, chatMic, sendButton} = this.args;

        // input focus disable(SendButton -> mic button)
        chatInput.blur(); 
        sendButton.classList.remove('active');
        chatMic.classList.add('active') 

        // get text in input
        let inputTxt = chatInput.value
        if (inputTxt === "") {
            return;
        }

        let msg1 = { name: "User", message: inputTxt }
        this.messages.push(msg1);

        getBotResponse(inputTxt);
    }
}


const chatbox = new Chatbox();
chatbox.display();

function getBotResponse(input, lang="en") {
    //let lang = "ar"; "en";
    // lang (input language)

    document.querySelector('.chat-inputbox').value = "";
    
    let state= null;
    let answer = "Hello";
    const token = localStorage.getItem('jwt-token') ;

    fetch("http://127.0.0.1:5000/respond", {
        method: 'POST',
        headers: {"Content-Type":
            "application/json", 'Authorization':
            'Bearer '+token},
        body: JSON.stringify({text: input, lang: lang, state: state})
    }).then(function(response) {
        if (!response.ok) throw Error(response.statusText)
        return response.json()
    }).then(function(text) {
        let answer = text.answer;
        let related_questions = text.related_questions;
        let pdfSrc = text.pdfSrc;
        let pdfTitle = text.pdfTitle;
        let videoSrc = text.videoSrc;
        console.log(videoSrc);

        var html = "";
        let msg2 = { name: "Sam", message: answer };
        chatbox.messages.push(msg2);

        if (related_questions != "") {
            var questionSplit = related_questions.split(";");
            for (var q in questionSplit) {
                var question = questionSplit[q];
                let msg3 = { name: "op", message: question };
                chatbox.messages.push(msg3);
            }
        } 

        var videoIndex = 0;
        if (videoSrc != "") {
            console.log('1');
            var videoSplit = videoSrc.split(";");
            for (var v in videoSplit) {
                videoIndex += 1;
                var video = videoSplit[v];
                let msg4 = { name: "video", message: video };
                chatbox.messages.push(msg4);
            }
            
            console.log('2');
        }
        var videoNum = videoIndex;

        chatbox.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "Sam")
            {
                html += '<div class="chat-message-bubble chat-message-bubble--bot"><div class="chat-message-default"><p>' + item.message + '</p></div></div>'
            }
            else if(item.name === "op") {
                html += '<a href="#" onclick="selectSuggest(this)" class="suggest-question"><span>' + item.message + '</span></a>'
            }
            else if(item.name === "video") {
                videoIndex -= 1;
                console.log('3');
                if (videoIndex == 0) {
                    html += '<div class="video-wrapper"><video><source src="' + item.message + '" type="video/mp4"></video></div></div>'
                } else if (videoIndex == videoNum - 1) {
                    html += '<div class="video-list"><div class="video-wrapper main-video"><video controls><source src="' + item.message + '" type="video/mp4"></video></div>'
                } else {
                    
                    console.log('5');
                    html += '<div class="video-wrapper"><video onclick="selectVideo(this)"><source src="' + item.message + '" type="video/mp4"></video></div>'
                }
            }
            else
            {
                html += '<div class="chat-message-bubble chat-message-bubble--user"><div class="chat-message-default"><p>' + item.message + '</p></div></div>'
            }
        });

        const chatmessage = document.querySelector('.chat-history');
        chatmessage.innerHTML = html;


        //getvoiceresponse(answer,lang);

        // getvideoresponse(answer, lang);
    }).catch(function(err) {
        window.alert('Error: ' + err.message);
    });
}

function selectSuggest(e) {
    e.preventDefault;
    document.querySelector('.chat-inputbox').value = e.innerText;
    chatbox.onSendButton();
}

// function getvoiceresponse(text,lang) {
//     let state= null;
//     const token = localStorage.getItem('jwt-token');
//     fetch("/getvoice", {
//         method: 'POST',
//         headers: {"Content-Type":
//             "application/json",
//             'Authorization': 'Bearer '+token},
//         body: JSON.stringify({text: text, lang: lang, state: state})
//     }).then(function(response) {
//         if (!response.ok) throw Error(response.statusText)
//         return response.blob()
//     }).then(function(blob) {

//     document.querySelector('#audio').src = URL.createObjectURL(blob); 
//     //text to speech reply, to be played directly
//     }).catch(function(err) {
//         window.alert('Error: ' + err.message); 
//     });
// }

//getvideoresponse
function getvideoresponse(text,lang) {
    let state= null;
    const token = localStorage.getItem('jwt-token');
    fetch("http://127.0.0.1:5000/getvideo", {
        method: 'POST',
        headers: {"Content-Type":
            "application/json",
            'Authorization': 'Bearer '+token},
        body: JSON.stringify({text: text, lang: lang, state: state})
    }).then(function(response) {
        if (!response.ok) throw Error(response.statusText)
        return response.blob()
    }).then(function(blob) {
        document.querySelector('#video').src = URL.createObjectURL(blob)


    }).catch(function(err) {
        window.alert('Error: ' + err.message);
    });
}

// function uploadFile(lang = "ar") {
//     let file = document.querySelector('#fileinput');
//     file = file.files[0];
//     console.log(file);
//     var form = new FormData();
//     const token = localStorage.getItem('jwt-token');

//     form.append('audio_data', file);
//     form.append('lang', lang);

//     const headers = new Headers({ 'Authorization': 'Bearer '+token});

//     fetch("/gettext", {
//         method: 'POST',
//         headers: headers,
//         cache: "no-cache",
//         body: form
//     }).then(function(response) {
//         if (!response.ok)
//             throw Error(response.statusText)
//         return (response.json())
//     }).then(function(text) {
//         respond_related_question(text);
//     }).catch(error => console.log(error));
// }

$('.control-overlay-mute button').click(function() {
    $(this).children('.btn-mute').toggleClass('hide');
    $(this).children('.btn-unmute').toggleClass('show');
});

