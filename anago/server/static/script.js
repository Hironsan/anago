function getMarkupText(val, data){
    var res = [];
    var dic = {"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "misc"};
    var words = val.split(" ");
    for (var i = 0; i < words.length; i++) {
        var w = words[i];
        for(key in data){
            if (data[key].indexOf(w) >= 0) {
                var text = '<small class="axlabel ' + dic[key] + '">' + w + '</small>';
                break;
            } else {
                var text = w;
            }
        }
        res.push(text);
    }
    return res.join(" ");
}

$(function () {
	$("#sentence").keypress(function (e) {
		var code = (e.keyCode ? e.keyCode : e.which);
		if (code == 13) {
		    var val = $("textarea#sentence").val();
		    console.log("value"+val);
		    $.post("/", {"sent": val},
			   function(data, status){
			       console.log(data);
			       var text = getMarkupText(val, JSON.parse(data));
			       $(".message-body").html(text);
			       console.log(text);
			});
		    //$("#submit").trigger('click');
		    return true;
		}
	    });
    });