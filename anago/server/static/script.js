function getMarkupText(data){
    var res = [];
    var dic = {"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "misc"};
    for (var i=0;i<data.length;i++) {
	var words = data[i][0];
	var tag = data[i][1];
	var chunk = words.join(" ");
	if (tag == 'O') {
	    text = chunk;
	} else {
	    var text = '<small class="axlabel ' + dic[tag] + '">' + chunk + '</small>';
	}
	res.push(text)
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
			       var text = getMarkupText(JSON.parse(data));
			       $(".message-body").html(text);
			       console.log(text);
			});
		    //$("#submit").trigger('click');
		    return true;
		}
	    });
    });