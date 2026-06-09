FROM caddy:2-builder AS builder

COPY caddy-jsonrpc/ /src/caddy-jsonrpc/

RUN xcaddy build \
    --with github.com/mholt/caddy-ratelimit \
    --with github.com/pearl-research-labs/pearl/proxy/caddy-jsonrpc=/src/caddy-jsonrpc

FROM caddy:2

COPY --from=builder /usr/bin/caddy /usr/bin/caddy
COPY Caddyfile /etc/caddy/Caddyfile
