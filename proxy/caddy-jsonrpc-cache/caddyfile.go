package jsonrpccache

import (
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
)

func init() {
	httpcaddyfile.RegisterHandlerDirective("jsonrpc_cache", parseCaddyfile)
	httpcaddyfile.RegisterDirectiveOrder("jsonrpc_cache", "before", "reverse_proxy")
}

// parseCaddyfile unmarshals tokens from h into a new Handler.
func parseCaddyfile(h httpcaddyfile.Helper) (caddyhttp.MiddlewareHandler, error) {
	var handler Handler
	err := handler.UnmarshalCaddyfile(h.Dispenser)
	return &handler, err
}

// UnmarshalCaddyfile implements caddyfile.Unmarshaler. Syntax:
//
//	jsonrpc_cache {
//	    cache <method> <ttl>
//	    miss_timeout <duration>
//	}
//
// `miss_timeout` is optional and bounds how long the cache may wait for a
// single upstream call before abandoning it. Without an explicit value the
// default (defaultMissTimeout) is used.
func (h *Handler) UnmarshalCaddyfile(d *caddyfile.Dispenser) error {
	for d.Next() {
		if d.NextArg() {
			return d.ArgErr()
		}

		for nesting := d.Nesting(); d.NextBlock(nesting); {
			switch d.Val() {
			case "cache":
				args := d.RemainingArgs()
				if len(args) != 2 {
					return d.ArgErr()
				}
				ttl, err := time.ParseDuration(args[1])
				if err != nil {
					return d.Errf("invalid TTL %q: %v", args[1], err)
				}
				h.Rules = append(h.Rules, CacheRule{
					Method: args[0],
					TTL:    caddy.Duration(ttl),
				})
			case "miss_timeout":
				args := d.RemainingArgs()
				if len(args) != 1 {
					return d.ArgErr()
				}
				timeout, err := time.ParseDuration(args[0])
				if err != nil {
					return d.Errf("invalid miss_timeout %q: %v", args[0], err)
				}
				if timeout < 0 {
					return d.Errf("miss_timeout must be non-negative, got %q", args[0])
				}
				h.MissTimeout = caddy.Duration(timeout)
			default:
				return d.Errf("unrecognized subdirective '%s'", d.Val())
			}
		}
	}
	return nil
}

// Interface guard
var _ caddyfile.Unmarshaler = (*Handler)(nil)
