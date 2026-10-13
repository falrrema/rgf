/* Return freed heap pages to the operating system; no effect on results. MIT license. */
#ifndef AZ_RELEASE_MEMORY_HPP
#define AZ_RELEASE_MEMORY_HPP

#include <cstdlib>
#if defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <malloc.h>
#endif

/*
 * The training input is held as many small blocks.  After it is destroyed the
 * allocator keeps those pages, so resident memory stays at the input's size
 * for the whole fit unless they are handed back.
 */
inline void AzReleaseFreedMemory() {
#if defined(__APPLE__)
  malloc_zone_pressure_relief(NULL, 0);
#elif defined(__GLIBC__)
  malloc_trim(0);
#endif
}

#endif
