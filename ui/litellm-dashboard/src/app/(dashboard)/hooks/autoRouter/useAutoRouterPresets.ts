import { AutoRouterPreset, hydratePresets } from "@/lib/autorouter_presets";
import { getAutoRouterPresets } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const presetKeys = createQueryKeys("autoRouterPresets");

export const useAutoRouterPresets = () => {
  // 1 hour, matching the proxy's own remote-catalog cache TTL: the catalog can change under a
  // running proxy (that is the point of serving it at runtime), so no release-length staleTime.
  const options = {
    queryKey: presetKeys.list({}),
    queryFn: async () => hydratePresets(await getAutoRouterPresets()),
    staleTime: 60 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  };
  return useQuery<AutoRouterPreset[]>(options);
};
