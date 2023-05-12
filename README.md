# Hallios Webapp - README

This README document provides information on how to serve the Hallios web application using Jekyll.

### Requirements

- Ruby 2.6 or higher
- RubyGems
- Jekyll
- Bundler

### Installation

To install Jekyll, first make sure you have Ruby and RubyGems installed. Then, open a terminal and run the following command:

```
gem install bundler jekyll
```

This will install Jekyll and all its dependencies.

## Serving the Webapp

Once you have installed Jekyll, you can serve the Hallios web application using the following command:

```
cd /path/to/hallios
bundle exec jekyll serve --host 0.0.0.0
```

This will start a local server and serve the Hallios web application at http://localhost:4000. You can access the web application using a web browser.

If you want to use livereload, you can add the --livereload option to the command:

```
cd /path/to/hallios
bundle exec jekyll serve --livereload --host 0.0.0.0
```

This will start a local server with livereload enabled.
=======
# hallios.github.io
hallios CTF team
