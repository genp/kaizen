

Setup

1. Install Ruby. 
	I think macs already have it. 

2. Install sass and compass. 
	You can just run "gem install compass", and I think sass will also be installed. If not, just run "gem install sass".

3. Double check and make sure everything is installed.
	You can run "sass -v" and "compass version". As of my writing this, the most current versions of sass is "Sass 3.4.0 (Selective Steve)", and I don't know for compass, but i've got "Compass 0.12.7 (Alnilam)""

4. Setting up a project.
	I don't think you'll actually have to do this, but i'm not completely sure. I just navigated to the top directory of our project and ran "compass create". This created a couple folders with intial sass and css files that I just deleted. More importantly, it created a config.rb file. In that file, there are some self explanatory fields that I changed for our project, most importantly the location of the sass files, and the location of the css files.

========================

Directories and compiling

5. app/static/sass
	I created a sass directory for us, which is inside the static directory. In it should go all of our scss files, which will get compiled into css files in the app/static/css directory. After making changes to the scss, i've just been typing "compass watch", which I think checks to see what's changed, and then creates the css. It actually continues to run and keep track of changes, so if you just leave that going while doing work, it will constantly keep updating the css. When the scss is compiled, it makes an equivalent file in the css directory, so something called test.scss becomes test.css

=========================

What you can do.

6. All the normal css stuff.
	Scss is just a superset of css. If you just were to write normal css in the .scss files, it would work fine. Most of the file is still just normal css.

7. Variables.
	You can define and use a variable using the $ sign. 

	Example:

		$main-color: red;
		$font-stack: "Nunito", sans-serif, Frutiger, "Frutiger Linotype", Univers, Calibri, "Gill Sans";
		$secondary-color: rgb(240,240,240);

		p {
			color: $main-color;
			font-family: $font-stack;
		}

		h1 {
			color: $secondary-color;
		}

8. Nesting.
	You can nest your css.

	Example:

		div {
			ul {
				width: 50px;
			}

			p {
				font-size: 15pt;
			}

			#item {
				color : red;
			}
		}

	Will compile to:

		div ul {
			width: 50px;
		}

		div p {
			font-size: 15pt;
		}

		div #item {
			color: red;
		}

9. Mixins.
	You can define a mixin with the @mixin tag, and use it with the @include tag.

	Example:

		@mixin border-radius($radius) {
	  		-webkit-border-radius: $radius;
	     	-moz-border-radius: $radius;
	      	-ms-border-radius: $radius;
	    	border-radius: $radius;
		}

		div { 
			@include border-radius(10px); 
		}


All that earlier stuff was just from sass. There's also more that I haven't used in our scss yet, like partials, inheritance, imports, and math operations. Examples and info on all of those, and basically all the examples I gave, can be found at: http://sass-lang.com/guide


There are also alot of things provided by using compass, but i've only used a few of them. Mostly, i've used the cross-browser css3 mixins it provides.

10. Compass mixins.
	You can import css mixins from compass, and use them.

	Example:

	@import "compass/css3"   

	div {
		@include border-radius(3px);
	}

	There are a ton more at http://compass-style.org/reference/compass/css3/



I think compass has alot more to offer, but I haven't used it much yet. There are a lot of good examples here: http://compass-style.org/examples/









