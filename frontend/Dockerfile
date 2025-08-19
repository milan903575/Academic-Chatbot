FROM nginx:alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Copy HTML templates
COPY templates/ /usr/share/nginx/html/

# Copy static assets
COPY static/ /usr/share/nginx/html/static/

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create necessary directories and set permissions
RUN mkdir -p /var/cache/nginx/client_temp \
    && chown -R nginx:nginx /usr/share/nginx/html \
    && chmod -R 755 /usr/share/nginx/html

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Expose port 80
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
